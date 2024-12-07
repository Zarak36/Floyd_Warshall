flowchart TB
    subgraph MainMemory["Main Memory Operations"]
        M[Distance Matrix in Shared Memory]
    end

    subgraph ThreadDataCreation["Thread Data Creation"]
        D[Create Thread Data Structures]
    end

    subgraph ThreadDataStructures["Thread Data Objects"]
        TD1[ThreadData 1<br/>start:0, end:1]
        TD2[ThreadData 2<br/>start:1, end:2]
        TD3[ThreadData 3<br/>start:2, end:3]
        TD4[ThreadData 4<br/>start:3, end:4]
    end

    subgraph WorkerThreads["Worker Threads"]
        W1[pthread 1<br/>Process rows 0-1]
        W2[pthread 2<br/>Process rows 1-2]
        W3[pthread 3<br/>Process rows 2-3]
        W4[pthread 4<br/>Process rows 3-4]
    end

    D --> TD1
    D --> TD2
    D --> TD3
    D --> TD4

    TD1 --> |"1. Pass ThreadData*"|W1
    TD2 --> |"1. Pass ThreadData*"|W2
    TD3 --> |"1. Pass ThreadData*"|W3
    TD4 --> |"1. Pass ThreadData*"|W4

    M --> |"2. Read Matrix Data"|W1
    M --> |"2. Read Matrix Data"|W2
    M --> |"2. Read Matrix Data"|W3
    M --> |"2. Read Matrix Data"|W4

    W1 --> |"3. Write Updated Distances"|M
    W2 --> |"3. Write Updated Distances"|M
    W3 --> |"3. Write Updated Distances"|M
    W4 --> |"3. Write Updated Distances"|M
