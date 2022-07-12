
/* Logestic map model: Network (Library and Regular)*/
//-----------------------------------------------------------------------------
//		Comments
//-----------------------------------------------------------------------------


#include "common_header.h"
#include "Def.h"
#include "Random.h"
#include "Read_parameters.h"
#include "Initialization.h"
#include "Find_cubic_hermite_root.h"
#include "Run_model.h"
#include "Delete.h"


int main(int argc,char **argv)
{
	long seed, seed0, seed1, seed2;
	clock_t t0, t1;	 
	char str[200];
	double MLE;
	double mean_fire_rate;
	bool verbose;
	// Config program options:
	po::options_description generic("Generic Options");
	generic.add_options()
		("help,h", "produce help message")
		("verbose,v", po::bool_switch(&verbose), "show output")
		("config,c", po::value<string>()->default_value("NetModel_parameters.ini"), "config filename.")
		;
	po::options_description config("Configs");
	config.add_options()
		("NE",          po::value<int>()->default_value(2), "num of E neurons")
		("NI",          po::value<int>()->default_value(0), "num of I neurons")
		("seed",        po::value<string>()->default_value("11 11"), "seed to generate connectivity matrix.")
		("T_Max",       po::value<double>()->default_value(1e7), "Simulation time period, unit ms.")
		("T_step",      po::value<double>()->default_value(0.2), "Time step, unit ms.")
		("S",           po::value<string>()->default_value("0.02 0.02 0.02 0.02"), "Synaptic coupling strength")
		("I_CONST",     po::value<double>()->default_value(0), "Constant external drive.")
		("P_c",         po::value<double>()->default_value(0.25), "Erdos-Renyi connecting probability.")
		("random_S",    po::value<int>()->default_value(0), "random mode of recurrent coupling strength (0-none 1-uniform 2-gauss 3-exponential 4-lognormal)")
		("Lyapunov",    po::value<int>()->default_value(0), "toggle to calculate Lyapunov exponent.")
		("record_spk",  po::value<int>()->default_value(1), "toggle to record spike train.")
		("record_v",    po::value<int>()->default_value(0), "toggle to record v.")
		("record_vlim", po::value<string>()->default_value("0 1e8"), "time range to record voltage trace.")
		("record_path", po::value<string>()->default_value("./data/"), "path to save data")
		;
	// create variable map
	po::variables_map vm;
	po::options_description cml_options;
	cml_options.add(generic).add(config);
	po::store(po::parse_command_line(argc, argv, cml_options), vm);
	po::notify(vm);
	if (vm.count("help")) {
		cout << generic << '\n';
		cout << config << '\n';
		return 1;
	}
	// loading parsers from config file
	ifstream config_file;
	if (vm.count("config")) {
		string cfname = vm["config"].as<string>();
		config_file.open(cfname.c_str());
		po::store(po::parse_config_file(config_file, config), vm);
		po::notify(vm);
	}
	// Override config params with cml params
	po::store(po::parse_command_line(argc, argv, cml_options), vm);
	po::notify(vm);

	vector<int> seed_buff;
	str2vec(vm["seed"].as<string>(), seed_buff);
	seed = seed_buff[0]; 
	seed1 = seed_buff[1];
	
	Read_parameters(vm);
	out_put_filename();
	seed0 = seed;    // Create connect matrix
	seed2 = seed1;  // Initialization
	Initialization(seed0, seed2);

	// EPSP
	//CS[0][1] = 0;
	//neu[1].x = neu[0].x;
	//neu[1].x += S[0];


	t0 = clock();
	Run_model();
	t1 = clock();


	double total_fire_num[2] = { 0 };
	for (int i = 0; i < N; i++)
		total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;

	
	mean_fire_rate = (total_fire_num[0] + total_fire_num[1]) / T_Max * 1000 / N; //(Hz)
	printf("mean firing rate = %0.2f(Hz)\n", mean_fire_rate);
	printf("Total time = %0.3fs \n\n", double(t1 - t0) / CLOCKS_PER_SEC);

	Delete();
	return 0;
}
