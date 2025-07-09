# `brain`: a simulated brain engine

## 1. overview

`brain` is a generic, high-performance simulation platform for computational pharmacology and neuroscience. It is designed to model the dynamic effects of psychoactive drugs on a simplified but biologically grounded model of the brain.

The design separates the *scientific model* from the *computation engine*:

-   **protocol (`protocol.yaml`):** A human-readable YAML file, supplied by the user, defining all system parameters: the brain's structure (regions, neurotransmitter systems, receptors), the properties of drugs, and the sequence of the entire experiment (e.g. stabilization, drug challenges, controlled withdrawal). This file is the "source code" of the scientific experiment.

-   **database (`drugs.parquet`):** A highly compressed, columnar data file containing the pharmacodynamic profiles (receptor affinities, efficacies) of various drugs. This is generated from raw data sources (BindingDB, ChEMBL) by the provided Python scripts (`scripts/{fetch,materialize}.py`).

-   **the engine:** A hyper-optimized, compiled Rust program, agnostic to the specific science being modeled. Its sole purpose is to read a protocol file and its associated data, and execute the defined experiment.

This architecture allows scientists to design and run complex *in silico* experiments by simply editing text files, without ever needing to recompile the underlying engine.

## 2. architecture

The engine is built on several key principles from high-performance computing (HPC) to enable large-scale, high-resolution simulations.

### 2.1. data-oriented design: flat struct of arrays (FSoA)

The most critical performance feature is the use of a *flat struct of arrays (FSoA)* memory layout. All dynamic properties for all synaptic units are stored in a single, contiguous `Vec<f32>`. Data for different properties of the same component are grouped together. For example, for a receptor with three properties (`activity`, `density`, `free fraction`) and `N` units, the memory looks like:

```
[A1..AN, D1..DN, F1..FN, (next component)...]
```

This layout guarantees that when the engine processes a specific property (like all receptor activities), the data is read sequentially from memory. This maximizes CPU cache locality and eliminates the cache misses that plague traditional object-oriented *array of structs (AoS)* layouts.

### 2.2. single instruction, multiple data (SIMD)

The SoA layout is designed explicitly to enable *SIMD (Single Instruction, Multiple Data)* vectorization. The engine leverages Rust's `portable_simd` feature to perform arithmetic on multiple data points simultaneously using wide vector registers (e.g. 256-bit AVX2). Instead of adding two numbers, it adds two vectors of 8 numbers in a single CPU instruction, providing a near-linear speedup for all core calculations.

### 2.3. compiled tick plan & fused kernels

A major source of slowdown in generic engines is dynamicism inside the hot loop—string comparisons, hash map lookups, and conditional branching. This engine eliminates all of them by performing a one-time *'compilation'* step in `Brain::new`.

1.  **Parsing:** It reads the declarative `protocol.yaml`.
2.  **Memory Mapping & Pointer Calculation:** It maps all named components (e.g. `"striatum.dopamine.d2"`) to offsets within the single `properties` buffer. It then calculates the *raw memory pointers* to the start of each property's data.
3.  **Instruction Generation:** It generates a `tick_plan` (a linear `Vec<Instruction>`). Each `Instruction` is a simple enum that encodes a high-level operation (e.g. `UpdateReceptor`, `UpdateSystem`) and contains only the pre-calculated raw pointers and parameters it needs to operate.

The `tick()` function, which runs 86,400 times per simulated day, is a "dumb" executor that makes two passes through the compiled plan. The architecture uses **fused kernels** to maximize performance:
-   **Pass 1 (Receptors):** It processes all receptor updates. For each receptor, it performs a complete, fused update for every synaptic unit: it calculates drug binding, NT binding, total activity, plasticity (down/upregulation), and presynaptic feedback all in one go, keeping all relevant data for that receptor hot in the cache.
-   **Pass 2 (Systems):** After all receptor feedback has been accumulated, it makes a final pass to update the neurotransmitter system levels.

This data-centric loop structure—iterating over all units for one instruction at a time—is the key to the engine's speed. All "thinking" (lookups, branching) is done once at the start.

## 3. the `protocol.yaml` file

This file, provided as the sole argument, is the user's complete interface to the engine.

-   **`globals`**: Defines simulation-wide parameters like `num_synaptic_units` (the resolution of the simulation) and the logging frequency.
-   **`drugs`**: Lists the drugs to be used and their pharmacokinetic (PK) parameters. The engine loads the pharmacodynamic (PD) profiles (receptor affinities) for these drugs from `drugs.parquet`.
-   **`components`**: The skeleton of the brain. This is a list where you declare every region, neurotransmitter system, and receptor population.
    -   **Hierarchy**: The `parent` key establishes the hierarchy (e.g. `d2` is a child of `dopamine`).
    -   **`receptor_type`**: An optional enum (`presynaptic` or `postsynaptic`) to define the receptor's function.
    -   **`params`**: A flexible map for defining biological properties (`synthesis`, `affinity`, `plasticity`, `feedback`). This allows you to encode specific neuroscience knowledge (e.g. D2 receptors have a higher plasticity rate in the striatum).
-   **`state`**: A dictionary of global variables, most importantly the initial dose of each drug.
-   **`sensors`**: A list of named metrics to be calculated. The `metric` field uses the format `"aggregator(expression)"`. The engine supports basic expressions, including multiplication, to derive biophysical quantities. For example:
    -   `"mean(striatum.serotonin.sert::density)"`
    -   `"1.0 - mean(striatum.serotonin.sert::free_fraction)"` (calculates percentage occupancy)
    -   `"mean(striatum.serotonin.sert::density * striatum.serotonin.sert::free_fraction)"` (calculates the absolute count of unbound receptors)
-   **`actuators` and `phases`**: Defines the experimental procedure as a sequence of `phases`. During each phase, `modulators` define the rules that govern the simulation's state, connecting `sensors` to `actuators` via a `ControlLaw` (e.g. `Fixed` value or `Pid` controller).