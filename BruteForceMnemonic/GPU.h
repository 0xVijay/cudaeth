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

#include <stdint.h>
#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void gl_bruteforce_mnemonic(
	const uint64_t* __restrict__ entropy,
	const tableStruct* __restrict__ tables,
	retStruct* __restrict__ ret
);

__global__ void gl_bruteforce_mnemonic_for_save(
	const uint64_t* __restrict__ entropy,
	const tableStruct* __restrict__ tables,
	retStruct* __restrict__ ret,
	uint32_t* __restrict__ hash160_ret,
	uint8_t* __restrict__ for_save
);


extern __constant__ uint32_t dev_num_bytes_find[1];
extern __constant__ uint32_t dev_generate_path[2];
extern __constant__ uint32_t dev_num_childs[1];
extern __constant__ uint32_t dev_num_paths[1];
extern __constant__ int16_t dev_static_words_indices[12];

// New allowlist constants
extern __constant__ uint32_t dev_use_allowlists[1];
extern __constant__ uint16_t dev_candidate_counts[12];
extern __constant__ uint16_t dev_candidate_indices[12][128]; // MAX_PER_POS = 128
extern __constant__ uint32_t dev_single_target_mode[1];
extern __constant__ uint8_t dev_target_address[20];
