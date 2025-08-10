/**
  ******************************************************************************
  * @author		Anton Houzich
  * @version	V2.0.0
  * @date		29-April-2023
  * @mail		houzich_anton@mail.ru
  * discussion  https://t.me/BRUTE_FORCE_CRYPTO_WALLET
  ******************************************************************************
  */


#include <stdafx.h>

#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <memory>
#include <sstream>
#include <iomanip>
#include <vector>
#include <map>
#include <omp.h>



#include "Dispatcher.h"
#include "GPU.h"
#include "KernelStride.hpp"
#include "Helper.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "../Tools/tools.h"
#include "../Tools/utils.h"
#include "../config/Config.hpp"

static std::thread save_thread;

int Generate_Mnemonic(void)
{



	cudaError_t cudaStatus = cudaSuccess;
	int err;
	ConfigClass Config;
	try {
		// Try config.json first, then fallback to config.cfg
		try {
			parse_config(&Config, "config.json");
		} catch (...) {
			parse_config(&Config, "config.cfg");
		}
		
		// Only process static words for legacy mode
		if (!Config.use_allowlists) {
			err = tools::stringToWordIndices(Config.static_words_generate_mnemonic + " ?", Config.words_indicies_mnemonic);
			if (err != 0)
			{
				std::cerr << "Error stringToWordIndices()!" << std::endl;
				return -1;
			}
		}
		
		uint64_t number_of_generated_mnemonics = (Config.number_of_generated_mnemonics / (Config.cuda_block * Config.cuda_grid)) * (Config.cuda_block * Config.cuda_grid);
		if ((Config.number_of_generated_mnemonics % (Config.cuda_block * Config.cuda_grid)) != 0) number_of_generated_mnemonics += Config.cuda_block * Config.cuda_grid;
		Config.number_of_generated_mnemonics = number_of_generated_mnemonics;
	}
	catch (...) {
		for (;;)
			std::this_thread::sleep_for(std::chrono::seconds(30));
	}

	devicesInfo();
	// Choose which GPU to run on, change this on a multi-GPU system.
	uint32_t num_device = 0;
#ifndef TEST_MODE
	std::cout << "\n\nEnter number of device: ";
	std::cin >> num_device;
#endif //GENERATE_INFINITY
	cudaStatus = cudaSetDevice(num_device);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return -1;
	}

	size_t num_wallets_gpu = Config.cuda_grid * Config.cuda_block;
	if (num_wallets_gpu < NUM_PACKETS_SAVE_IN_FILE)
	{
		std::cerr << "Error num_wallets_gpu < NUM_PACKETS_SAVE_IN_FILE!" << std::endl;
		return -1;
	}
	uint32_t num_bytes = 0;
	if (Config.chech_equal_bytes_in_adresses == "yes")
	{
#ifdef TEST_MODE
		num_bytes = 6;
#else
		num_bytes = 8;
#endif //TEST_MODE
	}

	std::cout << "\nNUM WALLETS IN ROUND GPU: " << tools::formatWithCommas(num_wallets_gpu) << std::endl << std::endl;
	DataClass* Data = new DataClass();
	KernelStrideClass* Stride = new KernelStrideClass(Data);
	size_t num_addresses_in_tables = 0;

	if (Config.use_allowlists && Config.single_target_mode) {
		// New mode: skip reading tables, set minimal values to avoid errors
		num_addresses_in_tables = 1;
		std::cout << "ALLOWLIST MODE: Skipping table loading, using single target comparison" << std::endl;
		
		// Set path configuration for new mode (m/44'/60'/0'/0/x with child 2 only)
		Config.generate_path[0] = 1;
		Config.generate_path[1] = 0;
		Config.num_child_addresses = 3; // Need 0,1,2 to get child 2
		Config.num_paths = 1;
	} else {
		std::cout << "READ TABLES! WAIT..." << std::endl;
		tools::clearFiles();
		err = tools::readAllTables(Data->host.tables, Config.folder_tables, "", &num_addresses_in_tables);
		if (err == -1) {
			std::cerr << "Error readAllTables!" << std::endl;
			goto Error;
		}

		if (num_addresses_in_tables == 0) {
			std::cerr << "ERROR READ TABLES!! NO ADDRESSES IN FILES!!" << std::endl;
			goto Error;
		}
	}

	if (Data->malloc(Config.cuda_grid, Config.cuda_block, Config.num_paths, Config.num_child_addresses, Config.save_generation_result_in_file == "yes" ? true : false) != 0) {
		std::cerr << "Error Data->Malloc()!" << std::endl;
		goto Error;
	}

	if (Stride->init() != 0) {
		std::cerr << "Error INIT!!" << std::endl;
		goto Error;
	}

	Data->host.freeTableBuffers();

	std::cout << "START GENERATE ADDRESSES!" << std::endl;
	std::cout << "PATH: " << std::endl;
	if (Config.use_allowlists && Config.single_target_mode) {
		std::cout << "m/44'/60'/0'/0/2 (single target mode)" << std::endl;
	} else {
		if (Config.generate_path[0] != 0) std::cout << "m/44'/60'/0'/0/0.." << (Config.num_child_addresses - 1) << std::endl;
		if (Config.generate_path[1] != 0) std::cout << "m/44'/60'/0'/1/0.." << (Config.num_child_addresses - 1) << std::endl;
	}

	std::cout << "\nGENERATE " << tools::formatWithCommas(Config.number_of_generated_mnemonics) << " MNEMONICS. " << tools::formatWithCommas(Config.number_of_generated_mnemonics * Data->num_all_childs) << " ADDRESSES. MNEMONICS IN ROUNDS " << tools::formatWithCommas(Data->wallets_in_round_gpu) << ". WAIT...\n\n";

	tools::generateRandomUint64Buffer(Data->host.entropy, Data->size_entropy_buf / (sizeof(uint64_t)));
	if (cudaMemcpyToSymbol(dev_num_bytes_find, &num_bytes, 4, 0, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cerr << "cudaMemcpyToSymbol to num_bytes_find failed!" << std::endl;
		goto Error;
	}
	if (cudaMemcpyToSymbol(dev_generate_path, &Config.generate_path, sizeof(Config.generate_path), 0, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cerr << "cudaMemcpyToSymbol to dev_generate_path failed!" << std::endl;
		goto Error;
	}
	if (cudaMemcpyToSymbol(dev_num_childs, &Config.num_child_addresses, 4, 0, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cerr << "cudaMemcpyToSymbol to dev_num_child failed!" << std::endl;
		goto Error;
	}
	if (cudaMemcpyToSymbol(dev_num_paths, &Config.num_paths, 4, 0, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cerr << "cudaMemcpyToSymbol to dev_num_paths failed!" << std::endl;
		goto Error;
	}
	if (cudaMemcpyToSymbol(dev_static_words_indices, &Config.words_indicies_mnemonic, 12 * 2, 0, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cerr << "cudaMemcpyToSymbol to dev_gen_words_indices failed!" << std::endl;
		goto Error;
	}
	
	// Transfer new allowlist constants
	uint32_t use_allowlists = Config.use_allowlists ? 1 : 0;
	if (cudaMemcpyToSymbol(dev_use_allowlists, &use_allowlists, 4, 0, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cerr << "cudaMemcpyToSymbol to dev_use_allowlists failed!" << std::endl;
		goto Error;
	}
	
	if (Config.use_allowlists) {
		// Validate allowlist data before GPU transfer
		for (int i = 0; i < 12; i++) {
			if (Config.candidate_counts[i] == 0) {
				std::cerr << "Error: Position " << i << " has no candidate words" << std::endl;
				goto Error;
			}
			if (Config.candidate_counts[i] > MAX_PER_POS) {
				std::cerr << "Error: Position " << i << " has too many candidates (" 
						  << Config.candidate_counts[i] << " > " << MAX_PER_POS << ")" << std::endl;
				goto Error;
			}
		}
		
		if (cudaMemcpyToSymbol(dev_candidate_counts, &Config.candidate_counts, sizeof(Config.candidate_counts), 0, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			std::cerr << "cudaMemcpyToSymbol to dev_candidate_counts failed!" << std::endl;
			goto Error;
		}
		
		if (cudaMemcpyToSymbol(dev_candidate_indices, &Config.candidate_indices, sizeof(Config.candidate_indices), 0, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			std::cerr << "cudaMemcpyToSymbol to dev_candidate_indices failed!" << std::endl;
			goto Error;
		}
		
		uint32_t single_target = Config.single_target_mode ? 1 : 0;
		if (cudaMemcpyToSymbol(dev_single_target_mode, &single_target, 4, 0, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			std::cerr << "cudaMemcpyToSymbol to dev_single_target_mode failed!" << std::endl;
			goto Error;
		}
		
		if (Config.single_target_mode) {
			// Validate target address
			bool all_zero = true;
			for (int i = 0; i < 20; i++) {
				if (Config.target_address_bytes[i] != 0) {
					all_zero = false;
					break;
				}
			}
			if (all_zero) {
				std::cerr << "Error: Target address is all zeros" << std::endl;
				goto Error;
			}
			
			if (cudaMemcpyToSymbol(dev_target_address, &Config.target_address_bytes, 20, 0, cudaMemcpyHostToDevice) != cudaSuccess)
			{
				std::cerr << "cudaMemcpyToSymbol to dev_target_address failed!" << std::endl;
				goto Error;
			}
		}
	}

	for (size_t step = 0; step < Config.number_of_generated_mnemonics / (Data->wallets_in_round_gpu); step++)
	{
		tools::start_time();
		if (Config.save_generation_result_in_file == "yes") {
			if (Stride->start_for_save(Config.cuda_grid, Config.cuda_block) != 0) {
				std::cerr << "Error START!!" << std::endl;
				goto Error;
			}
		}
		else
		{
			if (Stride->start(Config.cuda_grid, Config.cuda_block) != 0) {
				std::cerr << "Error START!!" << std::endl;
				goto Error;
			}
		}

		tools::generateRandomUint64Buffer(Data->host.entropy, Data->size_entropy_buf / (sizeof(uint64_t)));

		if (save_thread.joinable()) save_thread.join();

		if (Config.save_generation_result_in_file == "yes") {
			if (Stride->end_for_save() != 0) {
				std::cerr << "Error END!!" << std::endl;
				goto Error;
			}
		}
		else
		{
			if (Stride->end() != 0) {
				std::cerr << "Error END!!" << std::endl;
				goto Error;
			}
		}

		if (Config.save_generation_result_in_file == "yes") {
			save_thread = std::thread(&tools::saveResult, (char *)Data->host.save, Data->size_save_buf);
		}

		tools::checkResult(Data->host.ret);

		float delay;
		tools::stop_time_and_calc_sec(&delay);
		std::cout << "\rGENERATE: " << tools::formatWithCommas((double)Data->wallets_in_round_gpu / delay) << " MNEMONICS/SEC AND "
			<< tools::formatWithCommas((double)(Data->wallets_in_round_gpu * Data->num_all_childs) / delay) << " ADDRESSES/SEC"
			<< " | SCAN: " << tools::formatPrefix((double)(Data->wallets_in_round_gpu * Data->num_all_childs * num_addresses_in_tables) / delay) << " ADDRESSES/SEC"
			<< " | ROUND: " << step;
	}

	std::cout << "\n\nEND!" << std::endl;
	if (save_thread.joinable()) save_thread.join();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return -1;
	}

	return 0;
Error:
	std::cout << "\n\nERROR!" << std::endl;
	if (save_thread.joinable()) save_thread.join();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return -1;
	}

	return -1;
}







