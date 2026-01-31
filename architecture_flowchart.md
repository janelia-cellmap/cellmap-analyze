# cellmap-analyze Architecture Flowchart

## High-Level Overview

```mermaid
flowchart LR
    subgraph Input["Input Layer"]
        direction TB
        CLI["CLI Command"]
        YAML[/"YAML Config"/]
    end

    subgraph Processing["Processing Layer"]
        P1["ConnectedComponents<br/>ContactSites<br/>Skeletonize<br/>FillHoles<br/>CleanConnectedComponents<br/>MorphologicalOperations"]
    end

    subgraph Analysis["Analysis Layer"]
        direction TB
        A1["Measure<br/>FitLinesToSegmentations<br/>AssignToCells"]
    end

    subgraph Compute["Distributed Compute Layer"]
        direction TB
        Dask["Dask Distributed"]
        Blocks["Block Partitioning"]
        Aggregate["Global Aggregation"]
    end

    subgraph Output["Output Layer"]
        direction TB
        ZarrOut[("Multiscale Zarr")]
        CSV[/"CSV Metrics"/]
    end

    CLI --> YAML
    YAML --> Processing
    YAML --> Analysis

    Processing --> Dask
    Analysis --> Dask

    Dask --> Blocks
    Blocks --> Aggregate

    Aggregate --> ZarrOut
    Aggregate --> CSV
```

## Detailed Processing Pipeline

```mermaid
flowchart TD
    subgraph Entry["Entry Point"]
        CLI["CLI Command<br/>(e.g., connected-components)"]
        Config[/"run-config.yaml"/]
    end

    subgraph Setup["Setup Phase"]
        RP["RunProperties"]
        ExecDir["Setup Execution Directory"]
        Logger["Configure Logging"]
    end

    subgraph Init["Initialization"]
        IDI1["ImageDataInterface<br/>(Input Dataset)"]
        IDI2["ImageDataInterface<br/>(Output Dataset)"]
        Mask["MasksFromConfig<br/>(Optional)"]
    end

    subgraph Blockwise["Blockwise Phase (Parallel)"]
        direction TB
        CreateBlocks["Create DaskBlocks"]
        ReadBlock["Read Block Data<br/>(to_ndarray_ts)"]
        Process["Process Block<br/>(threshold, smooth, label)"]
        WriteBlock["Write Blockwise Results"]
    end

    subgraph Global["Global Phase"]
        direction TB
        Collect["Collect Block Results"]
        BuildGraph["Build Equivalence Graph"]
        Relabel["Generate Global Relabeling"]
        Apply["Apply Relabeling"]
    end

    subgraph Output["Output"]
        Multiscale["Create Multiscale<br/>Pyramid (s0, s1, s2...)"]
        Metadata["Write OME-Zarr<br/>Metadata"]
        Final[("Final Dataset")]
    end

    CLI --> Config
    Config --> RP
    RP --> ExecDir
    ExecDir --> Logger
    Logger --> Init

    Init --> IDI1
    Init --> IDI2
    Init --> Mask

    IDI1 --> CreateBlocks
    CreateBlocks --> ReadBlock
    ReadBlock --> Process
    Mask --> Process
    Process --> WriteBlock

    WriteBlock --> Collect
    Collect --> BuildGraph
    BuildGraph --> Relabel
    Relabel --> Apply

    Apply --> Multiscale
    Multiscale --> Metadata
    Metadata --> Final
```

## Data Flow Through ImageDataInterface

```mermaid
flowchart LR
    subgraph Input["Data Source"]
        Zarr[("Zarr Store")]
        N5[("N5 Store")]
    end

    subgraph IDI["ImageDataInterface"]
        Props["Properties:<br/>- voxel_size<br/>- roi<br/>- offset<br/>- dtype<br/>- chunk_shape"]
        TS["to_ndarray_ts()<br/>(TensorStore)"]
        DS["to_ndarray_ds()<br/>(funlib)"]
    end

    subgraph Features["Features"]
        F1["Auto Retry<br/>Voxel Resampling<br/>Axis Swap (N5)"]
    end

    subgraph Out["Output"]
        NDArray["NumPy Array"]
    end

    Zarr --> IDI
    N5 --> IDI
    IDI --> Props
    Props --> TS
    Props --> DS
    TS --> F1
    F1 --> NDArray
    DS --> NDArray
```

## Processing Classes Relationships

```mermaid
flowchart TB
    subgraph Mixin["Base Mixin"]
        CCM["ComputeConfigMixin<br/>- num_workers<br/>- compute_args<br/>- retries"]
    end

    subgraph Segmentation["Segmentation Processing"]
        CC["ConnectedComponents<br/>Label connected regions"]
        CCC["CleanConnectedComponents<br/>Filter by volume"]
        CS["ContactSites<br/>Find organelle contacts"]
    end

    subgraph Morphology["Morphological Processing"]
        FH["FillHoles<br/>Fill interior voids"]
        MO["MorphologicalOperations<br/>Erosion/Dilation"]
        LWM["LabelWithMask<br/>Apply mask labels"]
    end

    subgraph Analysis["Analysis"]
        M["Measure<br/>Compute object metrics"]
        FL["FitLinesToSegmentations<br/>Fit geometric lines"]
        A2C["AssignToCells<br/>Map to cell IDs"]
        Skel["Skeletonize<br/>Extract skeletons"]
    end

    CCM --> CC
    CCM --> CCC
    CCM --> CS
    CCM --> FH
    CCM --> MO
    CCM --> LWM
    CCM --> M
    CCM --> FL
    CCM --> Skel

    CC --> CCC
    CC --> CS
    CC --> M
    M --> A2C
    CC --> Skel
    Skel --> FL
```

## Measure Analysis Flow

```mermaid
flowchart TD
    subgraph Input["Inputs"]
        Seg[("Segmentation<br/>Dataset")]
        Org1[("Organelle 1<br/>(Optional)")]
        Org2[("Organelle 2<br/>(Optional)")]
    end

    subgraph Blockwise["Blockwise Measurement"]
        CreateBlock["Create DaskBlock<br/>with padding"]
        Read["Read Block +<br/>Face Neighbors"]
        Compute["Compute per-object:<br/>- Volume<br/>- Surface Area<br/>- Center of Mass<br/>- Bounding Box"]
        Contact["Compute Contact<br/>Surface Areas<br/>(if organelles provided)"]
        Trim["Trim Border Voxels"]
    end

    subgraph Aggregate["Global Aggregation"]
        Merge["Merge per-ID metrics<br/>across blocks"]
        Validate["Validate<br/>consistency"]
    end

    subgraph Output["Output"]
        DF["Pandas DataFrame"]
        CSV[/"CSV Export"/]
        Write["Write to disk"]
    end

    Seg --> CreateBlock
    Org1 --> Contact
    Org2 --> Contact

    CreateBlock --> Read
    Read --> Compute
    Compute --> Contact
    Contact --> Trim
    Trim --> Merge
    Merge --> Validate
    Validate --> DF
    DF --> CSV
    DF --> Write
```

## Dask Block Processing Pattern

```mermaid
flowchart LR
    subgraph Dataset["Full Dataset"]
        D1["Block 0"]
        D2["Block 1"]
        D3["Block 2"]
        D4["Block N..."]
    end

    subgraph Parallel["Parallel Workers"]
        W1["Worker 1"]
        W2["Worker 2"]
        W3["Worker 3"]
        WN["Worker N"]
    end

    subgraph Results["Block Results"]
        R1["Result 0"]
        R2["Result 1"]
        R3["Result 2"]
        R4["Result N"]
    end

    subgraph Final["Final Output"]
        Agg["Aggregate/<br/>Relabel"]
        Out[("Output<br/>Dataset")]
    end

    D1 --> W1
    D2 --> W2
    D3 --> W3
    D4 --> WN

    W1 --> R1
    W2 --> R2
    W3 --> R3
    WN --> R4

    R1 --> Agg
    R2 --> Agg
    R3 --> Agg
    R4 --> Agg
    Agg --> Out
```

## CLI Commands Overview

```mermaid
flowchart TB
    subgraph CLI["CLI Entry Points"]
        direction LR
        cmd1["connected-components"]
        cmd2["contact-sites"]
        cmd4["measure"]
        cmd5["skeletonize"]
        cmd6["fill-holes"]
        cmd7["clean-connected-components"]
        cmd8["assign-to-cells"]
    end

    subgraph Classes["Processing Classes"]
        CC["ConnectedComponents"]
        CS["ContactSites"]
        M["Measure"]
        Skel["Skeletonize"]
        FH["FillHoles"]
        CCC["CleanConnectedComponents"]
        A2C["AssignToCells"]
    end

    cmd1 --> CC
    cmd2 --> CS
    cmd4 --> M
    cmd5 --> Skel
    cmd6 --> FH
    cmd7 --> CCC
    cmd8 --> A2C
```

---

## How to View These Diagrams

1. **GitHub**: Copy the mermaid code blocks into any GitHub markdown file - GitHub renders Mermaid natively

2. **Mermaid Live Editor**: Paste into https://mermaid.live/

3. **VS Code**: Install "Markdown Preview Mermaid Support" extension

4. **Documentation Sites**: Most modern doc generators (MkDocs, Docusaurus, etc.) support Mermaid
