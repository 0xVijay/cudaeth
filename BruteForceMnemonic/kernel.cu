/**
  ******************************************************************************
  * @author		Anton Houzich
  * @version	V2.0.0
  * @date		29-April-2023
  * @mail		houzich_anton@mail.ru
  * discussion  https://t.me/BRUTE_FORCE_CRYPTO_WALLET
  ******************************************************************************
  */
// Removed stdafx.h for platform independence
#include <cstdint>
#include <cstddef>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
// Removed synchapi.h (Windows-specific)

#include <Dispatcher.h>
#include <thread>

int main()
{
    
    Generate_Mnemonic();


    while (1) {
        std::this_thread::sleep_for(std::chrono::seconds(100));
    }

    return 0;
}

