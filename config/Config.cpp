/**
  ******************************************************************************
  * @author		Anton Houzich
  * @version	V2.0.0
  * @date		29-April-2023
  * @mail		houzich_anton@mail.ru
  * discussion  https://t.me/BRUTE_FORCE_CRYPTO_WALLET
  ******************************************************************************
  */
#include "Config.hpp"
#include <tao/config.hpp>
#include "../Tools/tools.h"

int check_config(ConfigClass& config)
{
	int num_paths = 0;
	if (config.path_m44h_60h_0h_0_x == "yes") {
		num_paths++;
		config.generate_path[0] = 1;
	}
	else if (config.path_m44h_60h_0h_0_x != "no") {
		std::cerr << "Error parse path_m44h_0h_0h_0_x. Please write \"yes\" or \"no\"" << std::endl;
		throw std::logic_error("error parse config file");
	}
	if (config.path_m44h_60h_0h_1_x == "yes") {
		num_paths++;
		config.generate_path[1] = 1;
	}
	else if (config.path_m44h_60h_0h_1_x != "no") {
		std::cerr << "Error parse path_m44h_0h_0h_1_x. Please write \"yes\" or \"no\"" << std::endl;
		throw std::logic_error("error parse config file");
	}
	if (config.chech_equal_bytes_in_adresses == "yes") {
	}
	else if (config.chech_equal_bytes_in_adresses != "no") {
		std::cerr << "Error parse chech_equal_bytes_in_adresses. Please write \"yes\" or \"no\"" << std::endl;
		throw std::logic_error("error parse config file");
	}
	if (config.save_generation_result_in_file == "yes") {
	}
	else if (config.save_generation_result_in_file != "no") {
		std::cerr << "Error parse save_generation_result_in_file. Please write \"yes\" or \"no\"" << std::endl;
		throw std::logic_error("error parse config file");
	}


	if (config.num_child_addresses > 0xFFFF)
	{
		std::cerr << "Error num_child. Please enter a number less than 65,535" << std::endl;
		throw std::logic_error("error parse config file");
	}
	if (config.number_of_generated_mnemonics > 18000000000000000000)
	{
		std::cerr << "Error number_of_generated_mnemonics. Please enter a number less than 18,000,000,000,000,000,000" << std::endl;
		throw std::logic_error("error parse config file");
	}

	config.num_paths = num_paths;

	// Validate and process new allowlist mode
	if (tools::validateAndProcessWordConstraints(&config) != 0) {
		throw std::logic_error("error validating word constraints");
	}

	return 0;
}


int parse_config(ConfigClass* config, std::string path)
{
	try {
		const tao::config::value v = tao::config::from_file(path);

		// Try new JSON format first
		if (v.find("wallet_type") != nullptr) {
			// New format with allowlists
			config->wallet_type = v.optional_string("wallet_type", "");
			config->mnemonic_length = v.optional_unsigned("mnemonic_length", 12);
			
			// Parse ethereum settings
			if (v.find("ethereum") != nullptr) {
				const auto& eth = v["ethereum"];
				config->ethereum.target_address = eth.optional_string("target_address", "");
				config->ethereum.derivation_path = eth.optional_string("derivation_path", "");
				config->ethereum.passphrase = eth.optional_string("passphrase", "");
			}
			
			// Parse word constraints
			if (v.find("word_constraints") != nullptr) {
				const auto& constraints = v["word_constraints"];
				for (const auto& constraint : constraints.get_array()) {
					WordConstraint wc;
					wc.position = constraint["position"].get_unsigned();
					const auto& words = constraint["words"].get_array();
					for (const auto& word : words) {
						wc.words.push_back(word.get_string());
					}
					config->word_constraints.push_back(wc);
				}
			}
			
			// Set defaults for new mode 
			config->cuda_grid = v.optional_unsigned("cuda_grid", 1024);
			config->cuda_block = v.optional_unsigned("cuda_block", 256);
			
			// Legacy fields optional in new format
			config->folder_tables = v.optional_string("folder_tables", "");
			config->number_of_generated_mnemonics = v.optional_unsigned("number_of_generated_mnemonics", 1000000);
			config->num_child_addresses = v.optional_unsigned("num_child_addresses", 10);
			config->path_m44h_60h_0h_0_x = v.optional_string("path_m44h_60h_0h_0_x", "yes");
			config->path_m44h_60h_0h_1_x = v.optional_string("path_m44h_60h_0h_1_x", "no");
			config->static_words_generate_mnemonic = v.optional_string("static_words_generate_mnemonic", "");
			config->chech_equal_bytes_in_adresses = v.optional_string("chech_equal_bytes_in_adresses", "no");
			config->save_generation_result_in_file = v.optional_string("save_generation_result_in_file", "no");
		}
		else {
			// Legacy format
			config->folder_tables = access(v, tao::config::key("folder_tables")).get_string();
			config->number_of_generated_mnemonics = access(v, tao::config::key("number_of_generated_mnemonics")).get_unsigned();
			config->num_child_addresses = access(v, tao::config::key("num_child_addresses")).get_unsigned();
			config->path_m44h_60h_0h_0_x = access(v, tao::config::key("path_m44h_60h_0h_0_x")).get_string();
			config->path_m44h_60h_0h_1_x = access(v, tao::config::key("path_m44h_60h_0h_1_x")).get_string();
			config->static_words_generate_mnemonic = access(v, tao::config::key("static_words_generate_mnemonic")).get_string();
			config->chech_equal_bytes_in_adresses = access(v, tao::config::key("chech_equal_bytes_in_adresses")).get_string();
			config->save_generation_result_in_file = access(v, tao::config::key("save_generation_result_in_file")).get_string();
			config->cuda_grid = access(v, tao::config::key("cuda_grid")).get_unsigned();
			config->cuda_block = access(v, tao::config::key("cuda_block")).get_unsigned();
		}

		return check_config(*config);
	}
	catch (std::runtime_error& e) {
		std::cerr << "Error parse config file " << path << " : " << e.what() << '\n';
		throw std::logic_error("error parse config file");
	}
	catch (...) {
		std::cerr << "Error parse config file, unknown exception occured" << std::endl;
		throw std::logic_error("error parse config file");
	}
	return 0;
}


