==PROF== Connected to process 32417 (/home/zhzhzoo/expert/bench_expert_tiny)
==PROF== Profiling "grouped_fused_down_kernel" - 0 (1/5): 0%....50%....100% - 8 passes
==PROF== Profiling "produce_grouped_fused_kernel" - 1 (2/5): 0%....50%....100% - 8 passes
==PROF== Profiling "grouped_fused_down_kernel" - 2 (3/5): 0%....50%....100% - 8 passes
==PROF== Profiling "produce_grouped_fused_kernel" - 3 (4/5): 0%....50%....100% - 8 passes
==PROF== Profiling "grouped_fused_down_kernel" - 4 (5/5): 0%....50%....100% - 8 passes
fmt=E4M3 hidden=7168 inter=2048 tokens=1 avg_ms=3.801306 tok/s=263.07 implied_BW=11.59 GB/s (1.2% of 936 GB/s)
==PROF== Disconnected from process 32417
[32417] bench_expert_tiny@127.0.0.1
  unnamed>::grouped_fused_down_kernel(PackedTileMatrix, const float *, __half *, int, int, int, int) (8, 1, 7)x(128, 1, 1), Context 1, Stream 13, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.24
    SM Frequency                    Ghz         2.61
    Elapsed Cycles                cycle    1,190,245
    Memory Throughput                 %         9.97
    DRAM Throughput                   %         4.99
    Duration                         us       456.10
    L1/TEX Cache Throughput           %        12.21
    L2 Cache Throughput               %         2.00
    SM Active Cycles              cycle   971,432.47
    Compute (SM) Throughput           %         5.60
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.09 full     
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                     56
    Registers Per Thread             register/thread              50
    Shared Memory Configuration Size           Kbyte          102.40
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block            8.46
    # SMs                                         SM              66
    Stack Size                                                 1,024
    Threads                                   thread           7,168
    # TPCs                                                        33
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                0.09
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 15.15%                                                                                          
          The grid for this launch is configured to execute only 56 blocks, which is less than the 66 multiprocessors   
          used. This can underutilize some multiprocessors. If you do not intend to execute this kernel concurrently    
          with other workloads, consider reducing the block size to have at least one block per multiprocessor or       
          increase the size of the grid to fully utilize the available hardware resources. See the Hardware Model       
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more      
          details on launch configurations.                                                                             

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block            9
    Block Limit Shared Mem                block           10
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           36
    Theoretical Occupancy                     %           75
    Achieved Occupancy                        %         8.34
    Achieved Active Warps Per SM           warp         4.00
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 88.88%                                                                                    
          The difference between calculated theoretical (75.0%) and measured achieved occupancy (8.3%) can be the       
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Local Speedup: 25%                                                                                       
          The 9.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (75.0%) is limited by the number of required      
          registers.                                                                                                    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle      232,890
    Total DRAM Elapsed Cycles        cycle   37,365,760
    Average L1 Active Cycles         cycle   971,432.47
    Total L1 Elapsed Cycles          cycle   78,506,196
    Average L2 Active Cycles         cycle      227,820
    Total L2 Elapsed Cycles          cycle   23,309,904
    Average SM Active Cycles         cycle   971,432.47
    Total SM Elapsed Cycles          cycle   78,506,196
    Average SMSP Active Cycles       cycle   971,412.06
    Total SMSP Elapsed Cycles        cycle  314,024,784
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 14.79%                                                                                          
          One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum   
          instance value is 18.11% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 14.79%                                                                                          
          One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum 
          instance value is 18.11% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 14.79%                                                                                          
          One or more L1 Slices have a much lower number of active cycles than the average number of active cycles.     
          Maximum instance value is 18.11% above the average, while the minimum instance value is 100.00% below the     
          average.                                                                                                      

  unnamed>::produce_grouped_fused_kernel(PackedTileMatrix, PackedTileMatrix, const __half *, float *, int, int, int, int) (16, 1, 7)x(128, 1, 1), Context 1, Stream 13, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.24
    SM Frequency                    Ghz         2.61
    Elapsed Cycles                cycle    9,426,070
    Memory Throughput                 %        17.33
    DRAM Throughput                   %         1.25
    Duration                         ms         3.61
    L1/TEX Cache Throughput           %        18.14
    L2 Cache Throughput               %         3.36
    SM Active Cycles              cycle 9,003,276.24
    Compute (SM) Throughput           %         9.85
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.34 full     
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                    112
    Registers Per Thread             register/thread              48
    Shared Memory Configuration Size           Kbyte          102.40
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block           16.66
    # SMs                                         SM              66
    Stack Size                                                 1,024
    Threads                                   thread          14,336
    # TPCs                                                        33
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                0.34
    -------------------------------- --------------- ---------------

    OPT   If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have at least two  
          blocks per multiprocessor (compared to the currently executed 1.7 blocks) This way, blocks that aren't        
          waiting for __syncthreads() can keep the hardware busy.                                                       

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block           10
    Block Limit Shared Mem                block            5
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           20
    Theoretical Occupancy                     %        41.67
    Achieved Occupancy                        %        14.13
    Achieved Active Warps Per SM           warp         6.78
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 66.08%                                                                                    
          The difference between calculated theoretical (41.7%) and measured achieved occupancy (14.1%) can be the      
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Local Speedup: 58.33%                                                                                    
          The 5.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (41.7%) is limited by the required amount of      
          shared memory.                                                                                                

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle       464,080
    Total DRAM Elapsed Cycles        cycle   295,950,336
    Average L1 Active Cycles         cycle  9,003,276.24
    Total L1 Elapsed Cycles          cycle   621,973,198
    Average L2 Active Cycles         cycle  1,729,700.71
    Total L2 Elapsed Cycles          cycle   184,598,088
    Average SM Active Cycles         cycle  9,003,276.24
    Total SM Elapsed Cycles          cycle   621,973,198
    Average SMSP Active Cycles       cycle  9,003,252.05
    Total SMSP Elapsed Cycles        cycle 2,487,892,792
    -------------------------- ----------- -------------

  unnamed>::grouped_fused_down_kernel(PackedTileMatrix, const float *, __half *, int, int, int, int) (8, 1, 7)x(128, 1, 1), Context 1, Stream 13, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.24
    SM Frequency                    Ghz         2.61
    Elapsed Cycles                cycle    1,190,425
    Memory Throughput                 %         9.98
    DRAM Throughput                   %         4.98
    Duration                         us       456.26
    L1/TEX Cache Throughput           %        12.22
    L2 Cache Throughput               %         2.00
    SM Active Cycles              cycle   970,711.24
    Compute (SM) Throughput           %         5.61
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.09 full     
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                     56
    Registers Per Thread             register/thread              50
    Shared Memory Configuration Size           Kbyte          102.40
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block            8.46
    # SMs                                         SM              66
    Stack Size                                                 1,024
    Threads                                   thread           7,168
    # TPCs                                                        33
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                0.09
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 15.15%                                                                                          
          The grid for this launch is configured to execute only 56 blocks, which is less than the 66 multiprocessors   
          used. This can underutilize some multiprocessors. If you do not intend to execute this kernel concurrently    
          with other workloads, consider reducing the block size to have at least one block per multiprocessor or       
          increase the size of the grid to fully utilize the available hardware resources. See the Hardware Model       
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more      
          details on launch configurations.                                                                             

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block            9
    Block Limit Shared Mem                block           10
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           36
    Theoretical Occupancy                     %           75
    Achieved Occupancy                        %         8.33
    Achieved Active Warps Per SM           warp         4.00
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 88.89%                                                                                    
          The difference between calculated theoretical (75.0%) and measured achieved occupancy (8.3%) can be the       
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Local Speedup: 25%                                                                                       
          The 9.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (75.0%) is limited by the number of required      
          registers.                                                                                                    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle      232,894
    Total DRAM Elapsed Cycles        cycle   37,379,072
    Average L1 Active Cycles         cycle   970,711.24
    Total L1 Elapsed Cycles          cycle   78,431,376
    Average L2 Active Cycles         cycle   231,802.83
    Total L2 Elapsed Cycles          cycle   23,301,936
    Average SM Active Cycles         cycle   970,711.24
    Total SM Elapsed Cycles          cycle   78,431,376
    Average SMSP Active Cycles       cycle   970,690.81
    Total SMSP Elapsed Cycles        cycle  313,725,504
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 14.84%                                                                                          
          One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum   
          instance value is 18.17% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 14.84%                                                                                          
          One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum 
          instance value is 18.17% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 14.84%                                                                                          
          One or more L1 Slices have a much lower number of active cycles than the average number of active cycles.     
          Maximum instance value is 18.17% above the average, while the minimum instance value is 100.00% below the     
          average.                                                                                                      

  unnamed>::produce_grouped_fused_kernel(PackedTileMatrix, PackedTileMatrix, const __half *, float *, int, int, int, int) (16, 1, 7)x(128, 1, 1), Context 1, Stream 13, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.24
    SM Frequency                    Ghz         2.60
    Elapsed Cycles                cycle    9,441,041
    Memory Throughput                 %        17.33
    DRAM Throughput                   %         1.25
    Duration                         ms         3.62
    L1/TEX Cache Throughput           %        18.14
    L2 Cache Throughput               %         3.37
    SM Active Cycles              cycle 9,000,869.82
    Compute (SM) Throughput           %         9.85
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.34 full     
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                    112
    Registers Per Thread             register/thread              48
    Shared Memory Configuration Size           Kbyte          102.40
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block           16.66
    # SMs                                         SM              66
    Stack Size                                                 1,024
    Threads                                   thread          14,336
    # TPCs                                                        33
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                0.34
    -------------------------------- --------------- ---------------

    OPT   If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have at least two  
          blocks per multiprocessor (compared to the currently executed 1.7 blocks) This way, blocks that aren't        
          waiting for __syncthreads() can keep the hardware busy.                                                       

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block           10
    Block Limit Shared Mem                block            5
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           20
    Theoretical Occupancy                     %        41.67
    Achieved Occupancy                        %        14.13
    Achieved Active Warps Per SM           warp         6.78
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 66.08%                                                                                    
          The difference between calculated theoretical (41.7%) and measured achieved occupancy (14.1%) can be the      
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Local Speedup: 58.33%                                                                                    
          The 5.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (41.7%) is limited by the required amount of      
          shared memory.                                                                                                

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle       464,114
    Total DRAM Elapsed Cycles        cycle   296,923,136
    Average L1 Active Cycles         cycle  9,000,869.82
    Total L1 Elapsed Cycles          cycle   621,989,826
    Average L2 Active Cycles         cycle  1,736,283.33
    Total L2 Elapsed Cycles          cycle   184,600,176
    Average SM Active Cycles         cycle  9,000,869.82
    Total SM Elapsed Cycles          cycle   621,989,826
    Average SMSP Active Cycles       cycle  9,000,845.45
    Total SMSP Elapsed Cycles        cycle 2,487,959,304
    -------------------------- ----------- -------------

  unnamed>::grouped_fused_down_kernel(PackedTileMatrix, const float *, __half *, int, int, int, int) (8, 1, 7)x(128, 1, 1), Context 1, Stream 13, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.24
    SM Frequency                    Ghz         2.60
    Elapsed Cycles                cycle    1,191,120
    Memory Throughput                 %         9.96
    DRAM Throughput                   %         4.98
    Duration                         us       457.06
    L1/TEX Cache Throughput           %        12.22
    L2 Cache Throughput               %         2.00
    SM Active Cycles              cycle   970,639.64
    Compute (SM) Throughput           %         5.60
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.09 full     
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                     56
    Registers Per Thread             register/thread              50
    Shared Memory Configuration Size           Kbyte          102.40
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block            8.46
    # SMs                                         SM              66
    Stack Size                                                 1,024
    Threads                                   thread           7,168
    # TPCs                                                        33
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                0.09
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 15.15%                                                                                          
          The grid for this launch is configured to execute only 56 blocks, which is less than the 66 multiprocessors   
          used. This can underutilize some multiprocessors. If you do not intend to execute this kernel concurrently    
          with other workloads, consider reducing the block size to have at least one block per multiprocessor or       
          increase the size of the grid to fully utilize the available hardware resources. See the Hardware Model       
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more      
          details on launch configurations.                                                                             

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block            9
    Block Limit Shared Mem                block           10
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           36
    Theoretical Occupancy                     %           75
    Achieved Occupancy                        %         8.34
    Achieved Active Warps Per SM           warp         4.00
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 88.88%                                                                                    
          The difference between calculated theoretical (75.0%) and measured achieved occupancy (8.3%) can be the       
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Local Speedup: 25%                                                                                       
          The 9.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (75.0%) is limited by the number of required      
          registers.                                                                                                    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle      232,894
    Total DRAM Elapsed Cycles        cycle   37,444,608
    Average L1 Active Cycles         cycle   970,639.64
    Total L1 Elapsed Cycles          cycle   78,556,370
    Average L2 Active Cycles         cycle   221,474.12
    Total L2 Elapsed Cycles          cycle   23,281,632
    Average SM Active Cycles         cycle   970,639.64
    Total SM Elapsed Cycles          cycle   78,556,370
    Average SMSP Active Cycles       cycle   970,619.22
    Total SMSP Elapsed Cycles        cycle  314,225,480
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 14.8%                                                                                           
          One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum   
          instance value is 18.15% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 14.8%                                                                                           
          One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum 
          instance value is 18.15% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 14.8%                                                                                           
          One or more L1 Slices have a much lower number of active cycles than the average number of active cycles.     
          Maximum instance value is 18.15% above the average, while the minimum instance value is 100.00% below the     
          average.                                                                                                      

