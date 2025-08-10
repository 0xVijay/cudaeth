/**
  ******************************************************************************
  * @author		Anton Houzich
  * @version	V2.0.0
  * @date		29-April-2023
  * @mail		houzich_anton@mail.ru
  * discussion  https://t.me/BRUTE_FORCE_CRYPTO_WALLET
  ******************************************************************************
  */
#pragma once
#include <string>
#include <vector>

#define MAX_PER_POS 128

struct WordConstraint {
	int position;
	std::vector<std::string> words;
};

struct EthereumSettings {
	std::string target_address = "";
	std::string derivation_path = "";
	std::string passphrase = "";
};

struct ConfigClass
{
public:
	std::string folder_tables = "";

	uint64_t number_of_generated_mnemonics = 0;
	uint64_t num_child_addresses = 0;

	std::string path_m44h_60h_0h_0_x = "";
	std::string path_m44h_60h_0h_1_x = "";

	uint32_t generate_path[2] = { 0 };
	uint32_t num_paths = 0;


	int16_t words_indicies_mnemonic[12] = { 0 };
	std::string static_words_generate_mnemonic = "";
	std::string chech_equal_bytes_in_adresses = "";
	std::string save_generation_result_in_file = "";

	uint64_t cuda_grid = 0;
	uint64_t cuda_block = 0;
	
	// New allowlist mode fields
	std::string wallet_type = "";
	uint32_t mnemonic_length = 12;
	EthereumSettings ethereum;
	std::vector<WordConstraint> word_constraints;
	
	// Processed allowlist data
	uint16_t candidate_counts[12] = { 0 };
	uint16_t candidate_indices[12][MAX_PER_POS] = { 0 };
	uint8_t target_address_bytes[20] = { 0 };
	bool use_allowlists = false;
	bool single_target_mode = false;
	uint64_t total_combinations = 0;
public:
	ConfigClass()
	{
	}
	~ConfigClass()
	{
	}
};


int parse_config(ConfigClass* config, std::string path);

