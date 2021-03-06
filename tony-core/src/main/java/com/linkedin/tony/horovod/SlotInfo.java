/**
 * Copyright 2021 LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
 * See LICENSE in the project root for license information.
 */
package com.linkedin.tony.horovod;

/**
 * Introduce SlotInfo class to wrap consensus info, which is needed by Horovod workers.
 * SlotInfo is provided by Horovod Driver, which collect built-in python script process's output.
 * SlotInfo including info is like as follows, more details can be found on horovod-on-tony proposal
 *
 * hostname     localhost:98
 * rank         0
 * localRank    0
 * crossRank    0
 * size         4
 * localSize    2
 * crossSize    2
 *
 */
public class SlotInfo {
    private String hostname;
    private int rank;
    private int localRank;
    private int crossRank;
    private int size;
    private int localSize;
    private int crossSize;

    public SlotInfo() {
    }

    public SlotInfo(String hostname, int rank, int localRank, int crossRank, int size, int localSize, int crossSize) {
        this.hostname = hostname;
        this.rank = rank;
        this.localRank = localRank;
        this.crossRank = crossRank;
        this.size = size;
        this.localSize = localSize;
        this.crossSize = crossSize;
    }

    public String getHostname() {
        return hostname;
    }

    public void setHostname(String hostname) {
        this.hostname = hostname;
    }

    public int getRank() {
        return rank;
    }

    public void setRank(int rank) {
        this.rank = rank;
    }

    public int getLocalRank() {
        return localRank;
    }

    public void setLocalRank(int localRank) {
        this.localRank = localRank;
    }

    public int getCrossRank() {
        return crossRank;
    }

    public void setCrossRank(int crossRank) {
        this.crossRank = crossRank;
    }

    public int getSize() {
        return size;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public int getLocalSize() {
        return localSize;
    }

    public void setLocalSize(int localSize) {
        this.localSize = localSize;
    }

    public int getCrossSize() {
        return crossSize;
    }

    public void setCrossSize(int crossSize) {
        this.crossSize = crossSize;
    }
}
