<!-- # MAPredict -->

![](model_parser/mapredict.png)

### A static analysis driven memory access prediction framework for modern CPUs.

### 1. Overview
    MAPredict, a static analysis driven framework that provides memory access prediction by gathering application and machines properties at compile time. MAPredict  invokes analytical model to predict LLC-DRAM traffic by combining the application model, themachine model, and user-provided hints for capturing dynamic information. MAPredict is tested on different micro-architectures of Intel and provided high accuracy for application with different access patterns.

#### 2. Organaisation of the Repository
    2.1 [applications]: Follow the installation procedure to get all the content of this folder.  
          2.1.1 [applications/memory_research_ornl/all_apps_experiments]:  TAU-PAPI script to generate LLC-DRAM traffic data for applications. scripts are also available for different micro-architechtures of Intel and also for OpenMP execution.
           2.1.2 [applications/memory_research_ornl/aspen_model_generation]: This folder contains all model generation source code (annotated source code) which is used by MAPredict to generate application model.
    2.2 [aspen]: this folder contains aspen code which is needed for MAPredict. Aspen abstract models enables the creation of performance models.
    2.3 [OpenARC-devel]: This folder contains jar files (compiled) of the OpenARC and COMPASS framework. MAPredict is built on top of COMPASS and OpenARC. However, it is not open source and for this reason, only the java binaries are provided.
    2.4 [model_parser]: This folder contains MAPredict's model parser that traverses the application and machine model and invokes appropriate analytical model.
    2.5 [models]: This folder contains the application and machine model which are passes to MAPredict for prediction.
    2.6 [scripts]: These scripts are used to compile MAPredict and invoke appropriate machine and application models.
    
    
    
    
    
    
    
    
    convention is: ./[MAPredict_binary] [application_model] [machine_model]
  
  
  
  
    2.1.1 GCC 9.1
    2.1.2 Java
  
    
    
  
