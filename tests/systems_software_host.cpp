// SPDX-License-Identifier: AGPL-3.0-or-later
// Actual portable GPU arithmetic/index helpers, exercised on the host. This is
// not a CUDA compiler test or proof of device execution/race freedom.
#include "archbench/systems_software_model.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

float native_library_input(std::size_t, unsigned);
namespace model = archbench::systems_software_model;
using Shape = std::array<int, 3>;
using Reference = std::complex<long double>;

std::size_t checks = 0, transforms = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

std::size_t volume(Shape shape) { return std::size_t(shape[0]) * shape[1] * shape[2]; }

std::array<std::size_t, 3> coordinates(std::size_t index, Shape shape) {
    std::array<std::size_t, 3> out{};
    for (int axis = 0; axis < 3; ++axis) {
        out[axis] = index % shape[axis];
        index /= shape[axis];
    }
    return out;
}

std::size_t independent_partner(std::size_t index, Shape shape) {
    const auto position = coordinates(index, shape);
    std::size_t result = 0;
    for (int axis = 2; axis >= 0; --axis)
        result = result * shape[axis] + (shape[axis] - position[axis]) % shape[axis];
    return result;
}

template<class T>
std::vector<model::complex_value<T>> axis_algorithm(std::vector<model::complex_value<T>> input,
                                                 Shape shape, int sign) {
    const std::size_t count = input.size();
    std::vector<model::complex_value<T>> other(count);
    std::size_t stride = 1;
    for (int length : shape) {
        std::vector<model::complex_value<T>> roots(length);
        for (int frequency = 0; frequency < length; ++frequency) {
            const long double angle = sign * 2 * std::acos(-1.0L) * frequency / length;
            roots[frequency] = {static_cast<T>(std::cos(angle)), static_cast<T>(std::sin(angle))};
        }
        if (length > 1 && model::power_of_two(length)) {
            std::vector<unsigned> used(count, 0);
            for (std::size_t index = 0; index < count; ++index) {
                const auto source = model::bit_reversed_axis_index(index, length, stride);
                require(source < count, "bit reversal outside allocation");
                require(source / volume(shape) == index / volume(shape), "bit reversal crosses batches");
                ++used[source];
                other[index] = input[source];
            }
            require(std::all_of(used.begin(), used.end(), [](unsigned n) { return n == 1; }),
                    "bit reversal is not a permutation");
            input.swap(other);
            for (int half_span = 1; half_span < length; half_span *= 2) {
                std::fill(used.begin(), used.end(), 0);
                for (std::size_t pair = 0; pair < count / 2; ++pair) {
                    const auto at = model::locate_butterfly(pair, length, stride, half_span);
                    require(at.low < count && at.high < count && at.low != at.high,
                            "butterfly pair outside allocation or overlapping itself");
                    require(at.low / volume(shape) == at.high / volume(shape), "butterfly crosses batches");
                    require(at.root >= 0 && at.root < length, "butterfly root outside table");
                    ++used[at.low]; ++used[at.high];
                    model::complex_value<T> low{}, high{};
                    model::butterfly(input[at.low], input[at.high], roots[at.root], low, high);
                    input[at.low] = low; input[at.high] = high;
                }
                require(std::all_of(used.begin(), used.end(), [](unsigned n) { return n == 1; }),
                        "butterfly stage does not partition all outputs exactly once");
            }
        } else if (length > 1) {
            for (std::size_t index = 0; index < count; ++index)
                other[index] = model::direct_axis_value(input.data(), roots.data(), index, length, stride);
            input.swap(other);
        }
        stride *= length;
    }
    return input;
}

template<class T>
Reference dft_bin(const std::vector<model::complex_value<T>>& input, Shape shape,
                  std::size_t batch, std::size_t frequency, int sign) {
    const auto f = coordinates(frequency, shape);
    Reference value{};
    for (std::size_t index = 0; index < volume(shape); ++index) {
        const auto position = coordinates(index, shape);
        long double phase = 0;
        for (int axis = 0; axis < 3; ++axis)
            phase += static_cast<long double>(f[axis]) * position[axis] / shape[axis];
        const long double angle = sign * 2 * std::acos(-1.0L) * phase;
        const auto sample = input[batch * volume(shape) + index];
        value += Reference(sample.x, sample.y) * Reference(std::cos(angle), std::sin(angle));
    }
    return value;
}

template<class T> void compare(model::complex_value<T> got, Reference expected) {
    const long double tolerance = (sizeof(T) == sizeof(float) ? 2e-5L : 2e-12L) *
                                  std::max(1.0L, std::abs(expected));
    require(std::abs(Reference(got.x, got.y) - expected) < tolerance, "axis transform differs from independent DFT");
}

template<class T> void check_shape(Shape shape) {
    constexpr std::size_t batches = 2;
    const std::size_t count = volume(shape) * batches;
    std::vector<model::complex_value<T>> input(count);
    for (std::size_t index = 0; index < count; ++index)
        input[index] = {static_cast<T>(model::input_value(index, 5)), static_cast<T>(model::input_value(index, 19))};
    for (int sign : {-1, 1}) {
        const auto result = axis_algorithm(input, shape, sign);
        for (std::size_t index = 0; index < count; ++index)
            compare(result[index], dft_bin(input, shape, index / volume(shape), index % volume(shape), sign));
        ++transforms;
    }

    // Real forward: compare every stored half-spectrum element, including the
    // boundary planes and odd fastest-axis lengths.
    for (std::size_t index = 0; index < count; ++index) input[index].y = 0;
    const auto forward = axis_algorithm(input, shape, -1);
    const auto packed_volume = volume(shape) / shape[0] * (shape[0] / 2 + 1);
    for (std::size_t batch = 0; batch < batches; ++batch) {
        for (std::size_t stored = 0; stored < packed_volume; ++stored) {
            const auto full = model::compact_to_full(stored, shape[0]);
            require(model::full_to_compact(full, shape[0]) == stored, "compact index round trip failed");
            compare(forward[batch * volume(shape) + full], dft_bin(input, shape, batch, full, -1));
        }
    }
    ++transforms;

    // Construct Hermitian full input independently from the GPU index helpers.
    std::vector<model::complex_value<T>> packed(packed_volume * batches), reconstructed(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto batch = index / volume(shape), local = index % volume(shape);
        const auto partner = independent_partner(local, shape);
        require(model::conjugate_index(local, shape[0], shape[1], shape[2]) == partner,
                "multidimensional conjugate index differs from coordinate model");
        const auto canonical = std::min(local, partner);
        input[index] = {static_cast<T>(model::input_value(canonical, unsigned(batch + 5))),
            static_cast<T>(local == partner ? 0 : model::input_value(canonical, unsigned(batch + 8)) *
                           (local < partner ? 1 : -1))};
        if (local % shape[0] <= std::size_t(shape[0] / 2))
            packed[batch * packed_volume + local / shape[0] * (shape[0] / 2 + 1) + local % shape[0]] = input[index];
    }
    for (std::size_t index = 0; index < count; ++index) {
        const auto from = model::locate_hermitian(index, volume(shape), packed_volume, shape[0], shape[1], shape[2]);
        require(from.index < packed.size(), "Hermitian source outside allocation");
        reconstructed[index] = packed[from.index];
        if (from.conjugate) reconstructed[index].y = -reconstructed[index].y;
        require(reconstructed[index].x == input[index].x && reconstructed[index].y == input[index].y,
                "compact Hermitian reconstruction changed a stored value");
    }
    const auto inverse = axis_algorithm(reconstructed, shape, 1);
    for (std::size_t index = 0; index < count; ++index) {
        const auto expected = dft_bin(input, shape, index / volume(shape), index % volume(shape), 1);
        compare(inverse[index], expected);
        require(std::abs(expected.imag()) < 1e-10L, "Hermitian inverse reference is not real");
    }
    ++transforms;
}

int main() {
    for (std::size_t index : {std::size_t(0), std::size_t(1), std::size_t(1000),
                             std::size_t(0xffffffffU), std::size_t(0x100000000ULL)})
        for (unsigned salt : {0U, 1U, 3U, 5U, 7U, 19U, 0xffffffffU})
            require(model::input_value(index, salt) == native_library_input(index, salt),
                    "software stored-input generator differs from actual native source");
    for (std::size_t index = 0; index < 10000; ++index)
        require(model::input_value(index, unsigned(index % 32)) == native_library_input(index, unsigned(index % 32)),
                "software stored-input generator differs from actual native source");
    const std::vector<Shape> shapes{{1,1,1}, {2,1,1}, {8,1,1}, {7,1,1}, {10,1,1},
        {4,8,1}, {5,3,1}, {3,4,1}, {4,2,2}, {3,2,2}, {5,3,2}, {1,4,3}};
    for (const auto shape : shapes) { check_shape<float>(shape); check_shape<double>(shape); }
    std::cout << "passed: " << transforms << " transforms; " << checks
              << " arithmetic/layout/contract checks (host execution only)\n";
}
