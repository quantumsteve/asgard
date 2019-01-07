#pragma once

#include "element_table.hpp"
#include "program_options.hpp"
#include <map>
#include "tensors.hpp"
#include <vector>

// -----------------------------------------------------------------------------
// connectivity
// this components's purpose is to define the connectivity between
// elements in the element_table
// -----------------------------------------------------------------------------

// FIXME need to determine which of these need to be
int get_1d_index(int const level, int const cell);
fk::matrix<int> make_1d_connectivity(int const num_levels);
using list_set = std::vector<fk::vector<int>>;
list_set make_connectivity(element_table table, int const num_dims,
                           int const max_level_sum, int const max_level_val,
                           bool const sort_connected = true);
