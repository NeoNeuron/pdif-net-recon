#ifndef __NPY_IO__
#define __NPY_IO__

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>

inline bool is_npy_file(const std::string& fname) {
	std::string suffix = fname.substr(fname.find_last_of(".") + 1);
	return (suffix == "npy");
}

inline bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}


// Function to load a .npy file
template <typename T>
std::vector<T> load_npy(const std::string& filename, std::vector<size_t>& shape) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file.");
    }

    // Read the magic string
    char magic[6];
    file.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Not a valid .npy file.");
    }

    // Read the version number
    uint8_t version[2];
    file.read(reinterpret_cast<char*>(version), 2);

    // Read the header length
    uint16_t header_len_v1;
    uint32_t header_len_v2;
    size_t header_len;
    if (version[0] == 1) {
        file.read(reinterpret_cast<char*>(&header_len_v1), 2);
        header_len = header_len_v1;
    } else if (version[0] == 2 || version[0] == 3) {
        file.read(reinterpret_cast<char*>(&header_len_v2), 4);
        header_len = header_len_v2;
    } else {
        throw std::runtime_error("Unsupported .npy version.");
    }

    // Read the header
    std::vector<char> header(header_len + 1);
    file.read(header.data(), header_len);
    header[header_len] = '\0';  // Null-terminate the header string

    // Parse the shape from the header
    std::string header_str(header.data());
    std::string shape_str = header_str.substr(header_str.find("'shape':") + 8);
    shape_str = shape_str.substr(1, shape_str.find(')') - 1);  // Extract content within the tuple
    shape.clear();

    std::istringstream shape_stream(shape_str);
    std::string dim;
    while (std::getline(shape_stream, dim, ',')) {
        // Trim any leading/trailing whitespace and parentheses
        dim.erase(std::remove_if(dim.begin(), dim.end(), ::isspace), dim.end());
        if (!dim.empty() && dim.front() == '(') dim.erase(dim.begin());  // Remove leading '('
        if (!dim.empty() && dim.back() == ')') dim.pop_back();            // Remove trailing ')'

        if (!dim.empty()) {
            // std::cout << "Parsing dimension: '" << dim << "'" << std::endl;  // Debugging output
            try {
                shape.push_back(std::stoul(dim));
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid dimension found in shape: " + dim);
            }
        }
    }

    // Read the binary data
    std::vector<T> data;
    size_t num_elements = 1;
    for (size_t s : shape) num_elements *= s;
    data.resize(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(T));

    file.close();
    return data;
}

// Function to load the header of a .npy file
inline std::ifstream load_npy_header(const std::string& filename, std::vector<size_t>& shape) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file.");
    }

    // Read the magic string
    char magic[6];
    file.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Not a valid .npy file.");
    }

    // Read the version number
    uint8_t version[2];
    file.read(reinterpret_cast<char*>(version), 2);

    // Read the header length
    uint16_t header_len_v1;
    uint32_t header_len_v2;
    size_t header_len;
    if (version[0] == 1) {
        file.read(reinterpret_cast<char*>(&header_len_v1), 2);
        header_len = header_len_v1;
    } else if (version[0] == 2 || version[0] == 3) {
        file.read(reinterpret_cast<char*>(&header_len_v2), 4);
        header_len = header_len_v2;
    } else {
        throw std::runtime_error("Unsupported .npy version.");
    }

    // Read the header
    std::vector<char> header(header_len + 1);
    file.read(header.data(), header_len);
    header[header_len] = '\0';  // Null-terminate the header string

    // Parse the shape from the header
    std::string header_str(header.data());
    std::string shape_str = header_str.substr(header_str.find("'shape':") + 8);
    shape_str = shape_str.substr(1, shape_str.find(')') - 1);  // Extract content within the tuple
    shape.clear();

    std::istringstream shape_stream(shape_str);
    std::string dim;
    while (std::getline(shape_stream, dim, ',')) {
        // Trim any leading/trailing whitespace and parentheses
        dim.erase(std::remove_if(dim.begin(), dim.end(), ::isspace), dim.end());
        if (!dim.empty() && dim.front() == '(') dim.erase(dim.begin());  // Remove leading '('
        if (!dim.empty() && dim.back() == ')') dim.pop_back();            // Remove trailing ')'

        if (!dim.empty()) {
            // std::cout << "Parsing dimension: '" << dim << "'" << std::endl;  // Debugging output
            try {
                shape.push_back(std::stoul(dim));
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid dimension found in shape: " + dim);
            }
        }
    }

    // return the binary file handler
    return file;
}


// Function to extract dtype of a .npy file
inline std::string check_npy_dtype(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file.");
    }

    // Read the magic string
    char magic[6];
    file.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Not a valid .npy file.");
    }

    // Read the version number
    uint8_t version[2];
    file.read(reinterpret_cast<char*>(version), 2);

    // Read the header length
    uint16_t header_len_v1;
    uint32_t header_len_v2;
    size_t header_len;
    if (version[0] == 1) {
        file.read(reinterpret_cast<char*>(&header_len_v1), 2);
        header_len = header_len_v1;
    } else if (version[0] == 2 || version[0] == 3) {
        file.read(reinterpret_cast<char*>(&header_len_v2), 4);
        header_len = header_len_v2;
    } else {
        throw std::runtime_error("Unsupported .npy version.");
    }

    // Read the header
    std::vector<char> header(header_len + 1);
    file.read(header.data(), header_len);
    header[header_len] = '\0';  // Null-terminate the header string

    // Parse the header to extract shape and data type
    std::string header_str(header.data());

    // Extract the data type
    std::string dtype_str = header_str.substr(header_str.find("'descr':") + 9);
    dtype_str = dtype_str.substr(0, dtype_str.find(","));  // Extract content within the quotes
    dtype_str.erase(std::remove(dtype_str.begin(), dtype_str.end(), '\''), dtype_str.end());

    return dtype_str;
}


// Function to save data to a .npy file
template <typename T>
void save_npy(const std::string& filename, const std::vector<T>& data, const std::vector<size_t>& shape) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file.");
    }

    // Write the magic string
    file.write("\x93NUMPY", 6);

    // Write the version number
    uint8_t version[2] = {1, 0};  // Using version 1.0
    file.write(reinterpret_cast<char*>(version), 2);

    // Create and write the header
    std::ostringstream header_oss;
    header_oss << "{'descr': '" << (sizeof(T) == 4 ? "<f4" : "<f8")
               << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        header_oss << shape[i];
        if (i < shape.size() - 1) {
            header_oss << ", ";
        }
    }
    header_oss << "), }";

    std::string header_str = header_oss.str();
    size_t header_len = header_str.size();
    size_t padding_len = 16 - (10 + header_len) % 16;  // Pad to 16 bytes
    header_str.append(padding_len, ' ');
    header_str.back() = '\n';  // The last character must be a newline

    uint16_t header_len_v1 = static_cast<uint16_t>(header_len + padding_len);
    file.write(reinterpret_cast<char*>(&header_len_v1), 2);
    file.write(header_str.c_str(), header_len_v1);

    // Write the binary data
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));

    file.close();
}

template <typename T>
std::ofstream save_npy_header(const std::string& filename, const std::vector<size_t>& shape) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file.");
    }

    // Write the magic string
    file.write("\x93NUMPY", 6);

    // Write the version number
    uint8_t version[2] = {1, 0};  // Using version 1.0
    file.write(reinterpret_cast<char*>(version), 2);

    // Create and write the header
    std::ostringstream header_oss;
    header_oss << "{'descr': '" << (sizeof(T) == 4 ? "<f4" : "<f8")
               << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        header_oss << shape[i];
        if (i < shape.size() - 1) {
            header_oss << ", ";
        }
    }
    header_oss << "), }";

    std::string header_str = header_oss.str();
    size_t header_len = header_str.size();
    size_t padding_len = 16 - (10 + header_len) % 16;  // Pad to 16 bytes
    header_str.append(padding_len, ' ');
    header_str.back() = '\n';  // The last character must be a newline

    uint16_t header_len_v1 = static_cast<uint16_t>(header_len + padding_len);
    file.write(reinterpret_cast<char*>(&header_len_v1), 2);
    file.write(header_str.c_str(), header_len_v1);

    // Return the binary file handler
    return file;
}

#endif __NPY_IO__
