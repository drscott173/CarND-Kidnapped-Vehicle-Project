/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *  Modified on: May 6, 2017
 *      Author: Tiffany Huang
 *      Student: Scott Penberthy  <scott.penberthy@gmail.com>
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <math.h>

#include "particle_filter.h"
using namespace std;

/**
 * A NOTE ON COORDINATE SYSTEMS
 *
 * The map coordinate system has an inverted Y, negative above the X axis, positive below.
 * Particles also have an inverted Y, negative above the X axis, positive below.
 * As a result, a positive particle heading will now turn clockwise, rotating toward negative Y.
 *
 */

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

	num_particles = 16;
	srand(42);                              // But of course, 42.

	default_random_engine gen;
	normal_distribution<double> N_x(0, std[0]);
	normal_distribution<double> N_y(0, std[1]);
	normal_distribution<double> N_theta(0, std[2]);

	for (int i=0; i < num_particles; i++) {
		Particle p;
		float nx = N_x(gen)*(i > 0);         // we should use the original
		float ny = N_y(gen)*(i > 0);         // reading we are given, without noise,
		float ntheta = N_theta(gen)*(i > 0); // as the first particle 0

		// Create a particle whose (x,y) and position
		// vary a bit according to Gaussian noise
		// around our original values.  Assume all are
		// equally likely (weight = 1.0).
		p.x = x + nx;
		p.y = y + ny;
		p.theta = theta + ntheta;
		p.weight = 1.0;

		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	//  Add measurements to each particle and add random Gaussian noise.
	default_random_engine gen;
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	// For reading convenience I'll use shorter variable names
	float dt = delta_t;
	float v = velocity;
	float thd = yaw_rate;

	// Use physics to predict where a particle's motion, moving at velocity v
	// along a heading th0, turning at thd radians per second, traveling
	// dt seconds from (x0,y0).
	for (Particle &p: particles) {
		float th0 = p.theta;
		float xf = p.x;
		float yf = p.y;
		float thf = th0;

		if (fabs(thd) > 0.00001) {
			// Turning
			thf += thd*dt;
			xf += (v/thd)*(sin(thf)-sin(th0));
			yf += (v/thd)*(cos(th0)-cos(thf));
		}
		else {
			// Constant heading
			xf += v*sin(thf);
			yf -= v*cos(thf);
		}

		// Update particle and include noise (uncertainty)
		p.x = xf + N_x(gen);
		p.y = yf + N_y(gen);
		p.theta = thf + N_theta(gen);
		p.weight = 1.0;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	LandmarkObs chosen;
	double best;
	bool first;

	// Loop through observations.  For each one, loop
	// through our *predicted* observations for all map
	// landmarks within range.  Find the closest match,
	// then call that landmark our own.
	for (LandmarkObs &obs: observations) {
		first = true;
		float x0 = obs.x;  // a register for speed
		float y0 = obs.y;
		for (LandmarkObs pred: predicted) {
			double d = dist(x0,y0,pred.x,pred.y);
			if (first) {
				best = d;
				chosen = pred;
				first = false;
			}
			else if (d < best) {
				best = d;
				chosen = pred;
			}
		}
		obs.id = chosen.id;
	}
}

vector<LandmarkObs> predict_observations(Particle p, double range, Map map_landmarks) {
	// Loop through all the map landmarks in map space.  For those within
	// a given range, estimate what we might observe from our
	// sensor -- a distance x,y in our particle's coordinate system,
	// assuming the particle is at our origin (x0,y0) in map space,
	// heading at th0.
	vector<LandmarkObs> predicted;
	float x0 = p.x;   // remember our particle info for speed
	float y0 = p.y;
	float th0 = p.theta;
	for (Map::single_landmark_s m: map_landmarks.landmark_list) {
		float dx = m.x_f - x0;
		float dy = m.y_f - y0;
		float r = sqrt(dx*dx+dy*dy);

		if (r < range) {
			// Within range, estimate what we should observe
			// for this landmark in local particle space.
			LandmarkObs obs;
			float beta = atan2(-dy, dx);  // angle of vector in map space
			float alpha = beta+th0;       // angle of vector in particle space
			obs.x = r*cos(alpha);         // guess x,y in particle space
			obs.y = -r*sin(alpha);
			obs.id = m.id_i;              // remember our landmark
			predicted.push_back(obs);
		}
	}
	return predicted;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	// more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	double total_p = 0.0;            // total, relative probability mass for a particle
	float sx2 = std_landmark[0];     // noise in x
	float sy2 = std_landmark[1];     // noise in y
	float a = 1.0/(2.0*M_PI*sx2*sy2);  // scalar for multivariate guassian

	sx2 = 2*sx2*sx2; // now squared, doubled for our multivariate Gaussian
	sy2 *= 2*sy2*sy2;

	for (Particle &p: particles) {
		vector<LandmarkObs> predicted;

		// associate observations with landmarks
		predicted = predict_observations(p, sensor_range, map_landmarks);
		dataAssociation(predicted, observations);

		// compute probability this particle is correct, initially 1.0
		float prob = 1.0;
		float x0 = p.x;
		float y0 = p.y;
		float th0 = p.theta;

		for (Map::single_landmark_s m: map_landmarks.landmark_list) {
			for (LandmarkObs obs: observations) {
				if (obs.id == m.id_i) {
					// determine the probability that map landmark m
					// actually could be at location (xm, ym), which
					// we estimate from an observation obs taken by
					// particle p at (x0,y0) with heading th0.

					// map from particle to map coordinates.
					double r = sqrt(obs.x*obs.x + obs.y*obs.y);  // dist to particle origin
					double beta = atan2(obs.y, obs.x) + th0; // angle from origin to landmark, map orientation
					double xm = x0 + r*cos(beta); // a vector of length r at angle beta
					double ym = y0 + r*sin(beta);

					// figure out the probability of landmark position xm,ym using
					// a 2-dimensional bayesian probability distribution
					double dx = (xm - m.x_f);
					double dy = (ym - m.y_f);
					double pxy = a * exp(-1.0*(((dx*dx)/sx2) + ((dy*dy)/sy2)));

					// combine this probability into total probability for
					// all measurements
					pxy = min(1.0, pxy);  // odd high probabilities when close to mean!
					prob *= pxy;
				}
			}
		}

		// store our probability as our new particle weight
		p.weight = prob;
		total_p += prob;
	}

	// NOTE: We skip normalization for speed.  Why?  The absolute probabilities
	// are sufficient to create a weighting vector for a discrete distribution.

}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	vector<Particle> sample;
	vector<double> weights;
	std::default_random_engine gen;

	// create a weighting vector for a discrete distribution
	for (Particle p: particles) {
		weights.push_back((double) 100*p.weight);
	}

	// Use a discrete distribution to pick particles and copy them
	// into a new selection.
	discrete_distribution<int> d(weights.begin(), weights.end());
	for (int i=0; i < num_particles; i++) {
		int index = d(gen);                // pick next random index
		Particle p, p0 = particles[index]; // copy the chosen particle
		p.x = p0.x;
		p.y = p0.y;
		p.weight = 1.0;
		p.theta = p0.theta;
		sample.push_back(p);
	}
	particles = sample;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
