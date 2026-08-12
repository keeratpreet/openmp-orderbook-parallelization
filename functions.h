#pragma once

// OpenMP-parallelized order-book processing over a stream of bit-stuffed
// exchange packets. See functions.cpp for the packet layout/decoding and the
// parallelization strategy behind each function.

#include <iostream>
#include <string>
#include <cstdint>
#include <vector>
#include <iomanip>
#include <fstream>
#include <map>
#include <algorithm>
#include <cmath>
#include <unordered_map>

// Replays orderBook sequentially, writing a sorted per-stock snapshot
// (snap_<n>.txt) every `freq` orders.
void updateDisplay(const std::vector<uint64_t> &orderBook, int32_t freq);

// Sum of (qty * value) across every order in orderBook.
int64_t totalAmountTraded(const std::vector<uint64_t> &orderBook);

// Writes per-stock aggregates (min sell, max buy, average order value) to stats.txt.
void printOrderStats(const std::vector<uint64_t> &orderBook);