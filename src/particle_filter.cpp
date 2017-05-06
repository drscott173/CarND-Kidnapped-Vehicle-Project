/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <math.h>
#define DEBUG 0
#define TRACE 0

#include "particle_filter.h"
using namespace std;

void ParticleFilter::show_particles() {
	return;
	float pi = M_PI;
	for (Particle p: particles) {
		cout << " [" << p.x << ", " << p.x << ", " << p.theta << "] ";
		cout << p.weight*100 << "% " << endl;
	}
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	srand(42693);

	default_random_engine gen;
	normal_distribution<double> N_x(0, std[0]);
	normal_distribution<double> N_y(0, std[1]);
	normal_distribution<double> N_theta(0, std[2]);

	for (int i=0; i < num_particles; i++) {
		Particle p;
		float nx = N_x(gen)*(i > 0);
		float ny = N_y(gen)*(i > 0);
		float ntheta = N_theta(gen)*(i > 0);

		p.x = x + nx;
		p.y = y + ny;
		p.theta = theta + ntheta;
		p.weight = 1.0;
		particles.push_back(p);
	}
	is_initialized = true;

	//cout << "Particles init:" << endl;
	show_particles();

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	// For reading convenience I'll use shorter variable names
	float dt = delta_t;
	float v = velocity;
	float thd = yaw_rate;

	if (TRACE) {
		cout << "dt " << dt << endl;
		cout << "v " << v << endl;
		cout << "thd " << thd << endl;
	}

	for (Particle &p: particles) {
		float th0 = p.theta;
		float xf = p.x;
		float yf = p.y;
		float thf = th0;

		if (DEBUG) {
			cout << "Moving [" << xf << ", " << yf << ", "<< thf << "] to ";
		}

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

		// Update particle and include noise
		p.x = xf + N_x(gen);
		p.y = yf + N_y(gen);
		float dth = N_theta(gen);
		p.theta = thf + dth;
		p.weight = 1.0;
		if (DEBUG) {
			cout << "[" << p.x << ", " << p.y << ", " << p.theta << "]" << endl;
			cout << "... with dth=" << dth << endl;
		}
	}

	show_particles();
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	LandmarkObs chosen;
	double best;
	bool first = true;

	if (DEBUG) {
		cout << "Comparing " << observations.size() << " obs with " << predicted.size() << " preds" << endl;
	}

	for (LandmarkObs &obs: observations) {
		first = true;
		float x0 = obs.x;
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
		if (DEBUG) {
			cout << " observed map point #" << obs.id << " at ";
			cout << "(" << x0 << ", " << y0 << ") with d=" << best << endl;
		}
	}
}

vector<LandmarkObs> ParticleFilter::predict_observations(Particle p, double range, Map map_landmarks) {
	// predict what we should be observing at particle p
	// given a set of map landmarks.  these observations are
	// all in the local coordinate system of p at (x0, y0) heading th0,
	// given map landmarks in a map coordinate system (x_f, y_f).
	vector<LandmarkObs> predicted;

	float x0 = p.x;
	float y0 = p.y;
	float th0 = p.theta;
	for (Map::single_landmark_s m: map_landmarks.landmark_list) {
		LandmarkObs obs;

		obs.id = m.id_i;

		// first absolute differences at orientation = 0
		float pi = M_PI;
		float dx = m.x_f - x0;
		float dy = m.y_f - y0;
		float r = sqrt(dx*dx+dy*dy);

		if (r < range) {
			float beta;  // angle of vector in map coordinate system
			if (fabs(dx) < 0.00001) {
				beta = pi/2.0;
				if (dy < 0) {
					beta = -beta;
				}
			}
			else {
				beta = atan2(-dy,dx);
			}
			float alpha = beta+th0;  // angle of vector in particle coordinate system, angles grow cw not ccw

			obs.x = r*cos(alpha); // x,y in particle coordinate system
			obs.y = -r*sin(alpha);

			if (DEBUG) {
				cout << " predict m[" << m.id_i << "] = (" << dx << ", " << dy << ") from ";
				cout << "(" << x0 << ", " << y0 << ") to ";
				cout << "(" << m.x_f << ", " << m.y_f << ")" << endl;
				cout << "beta " << beta << " alpha " << alpha << endl;
				cout <<" predicting observation of #" << obs.id << " at (" << obs.x << ", " << obs.y << ") at ";
				cout << th0 << " with d=" << r << endl;
			}
			predicted.push_back(obs);
		}
	}
	if (DEBUG) {
		cout << "..found " << predicted.size() << " navigation points" << endl;
	}

	return predicted;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	if (DEBUG) {
		cout << "Updating weights " << endl;
	}

	default_random_engine gen;
	normal_distribution<double> N_x(0, std_landmark[0]);
	normal_distribution<double> N_y(0, std_landmark[1]);

	double total_p = 0.0;
	float pi = M_PI;
	float sx = std_landmark[0];
	float sy = std_landmark[1];
	float a = 1.0/(2.0*pi*sx*sy);

	if (DEBUG) {
		cout << "Now have " << particles.size() << " particles" << endl;
		cout << "With scalar a=" << a << endl;
	}

	for (Particle &p: particles) {
		vector<LandmarkObs> predicted;
		if (DEBUG) {
			cout << endl;
			cout << "Processing P=(" << p.x << ", " << p.y << ", " << p.theta << ")" << endl;
		}

		// associate observations with landmarks
		predicted = predict_observations(p, sensor_range, map_landmarks);
		dataAssociation(predicted, observations);

		// compute probability this particle is correct, initially 1.0
		float prob = 1.0;
		float x0 = p.x;
		float y0 = p.y;
		float th0 = p.theta;

		if (DEBUG) {
			cout << "Weights for P=(" << x0 << ", " << y0 << ", " << th0 << ")" << endl;
		}
		for (Map::single_landmark_s m: map_landmarks.landmark_list) {
			for (LandmarkObs obs: observations) {
				if (obs.id == m.id_i) {
					// determine the probability that map landmark m
					// actually could be at location (xm, ym), which
					// we estimate from an observation obs taken by particle p.
					// here we map from particle to map coordinates (xm, ym);
					double r = sqrt(obs.x*obs.x + obs.y*obs.y);  // dist to particle origin
					double beta = atan2(obs.y, obs.x) + th0; // angle from origin to landmark, map coordinates
					double xm = x0 + r*cos(beta);
					double ym = y0 + r*sin(beta);
					double dx = (xm - m.x_f);
					double dy = (m.y_f - ym);

					// figure out the probability of our position xm,ym using
					// a 2-dimensional bayesian probability distribution
					double pxy = a * exp(-1.0*(((dx*dx)/(2*sx*sx)) + ((dy*dy)/(2*sy*sy))));

					// combine this probability into total probability for
					// all measurements
					//if (fabs(pxy) > 0.00001) {
					if (DEBUG) {
						cout << " o[" << obs.id << "]=" << pxy;
						cout << " guessing (" << xm << ", " << ym << ") vs ";
						cout << "(" << m.x_f << ", " << m.y_f << ")" << endl;
						cout << " r " << r << " beta " << beta << " th0 " << th0 << endl;
					}
						prob *= pxy;
					//}
				}
			}
		}

		// store our probability as our new particle weight
		// do we need to normalize these?
		p.weight = prob;
		if (DEBUG) {
			cout << " Final prob=" << prob << endl;
		}
		total_p += prob;
	}

	if (TRACE) {
		cout << "Found total prob " << total_p << ", normalizing." << endl;
	}
	if (fabs(total_p) > 0.0001) {
		double scalar = 1.0/total_p;
		for (Particle &p: particles) {
			p.weight = p.weight * scalar;
			if (DEBUG) {
				cout << "  P(" << p.x << ", " << p.y << ", " << p.theta << ")=";
				cout << p.weight << endl;
			}
		}
	}
	if (DEBUG) {
		cout << "Done normalizing" << endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> sample;
	vector<double> weights;

	// find max weight
	float wmax = 0;
	for (Particle p: particles) {
		if (p.weight > wmax) {
			wmax = p.weight;
		}
		weights.push_back((double) 100*p.weight);
	}

	std::default_random_engine gen;
    discrete_distribution<int> d(weights.begin(), weights.end());

    for (int i=0; i < num_particles; i++) {
    	Particle p;
    	int index = d(gen);

    	Particle p0 = particles[index];

    	p.x = p0.x;
    	p.y = p0.y;
    	p.weight = 1.0;
    	p.theta = p0.theta;
    	if (DEBUG) {
    		cout << "chose #" << index << " (" << p.x << ", " << p.y << ", " << p.theta << ")" << endl;
    	}
    	sample.push_back(p);
    }
    particles = sample;

    /**
    std::map<int, int> m;
    for(int n=0; n<10000; ++n) {
        ++m[d(gen)];
    }
    for(auto p : m) {
        std::cout << p.first << " generated " << p.second << " times\n";
    }
	default_random_engine gen;
	normal_distribution<double> N_beta(wmax, wmax/2.0);

	// around and around we go, sampling from the circle of weights
	int index = 0;
	double beta = 0.0;
	Particle p_i = particles[index];
	wmax *= 2.0;

	for (int i=0; i < num_particles; i++) {
		Particle p;
		double w_random = wmax*(((double) (rand() % 10000))/10000.0);
		beta += w_random; //N_beta(gen);

		while (p_i.weight < beta) {
			if (DEBUG) {
				cout << "Skipping " << index << " with B " << beta;
				cout << ", p[" <<  index << "]=" << p_i.weight << endl;
			}
			beta -= p_i.weight;
			index = (index+1)%num_particles;
			p_i = particles[index];
		}

		p.x = p_i.x;
		p.y = p_i.y;
		p.theta = p_i.theta;
		//beta -= p_i.weight;
		if (1 || DEBUG) {
			cout << "Chose P_" << index;
			cout << " = [" << p.x << ", " << p.y << ", " << p.theta << "]";
			cout << " with B=" << beta << ", r=" << w_random << endl;
		}
		sample.push_back(p);
	}

	// update our particle list
	particles = sample;
		**/

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
