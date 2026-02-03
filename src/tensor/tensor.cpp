#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace {
size_t numel_from_shape(const std::vector<size_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

bool compute_view_strides(
    const std::vector<size_t> &old_shape,
    const std::vector<ptrdiff_t> &old_strides,
    const std::vector<size_t> &new_shape,
    std::vector<ptrdiff_t> &new_strides) {
    new_strides.assign(new_shape.size(), 0);
    size_t old_numel = numel_from_shape(old_shape);
    size_t new_numel = numel_from_shape(new_shape);
    if (old_numel != new_numel) {
        return false;
    }
    if (new_numel == 0) {
        ptrdiff_t stride = 1;
        for (size_t i = new_shape.size(); i-- > 0;) {
            new_strides[i] = stride;
            stride *= static_cast<ptrdiff_t>(new_shape[i]);
        }
        return true;
    }

    std::vector<size_t> compact_shape;
    std::vector<ptrdiff_t> compact_strides;
    compact_shape.reserve(old_shape.size());
    compact_strides.reserve(old_strides.size());
    for (size_t i = 0; i < old_shape.size(); i++) {
        if (old_shape[i] != 1) {
            compact_shape.push_back(old_shape[i]);
            compact_strides.push_back(old_strides[i]);
        }
    }

    if (compact_shape.empty()) {
        ptrdiff_t stride = 1;
        for (size_t i = new_shape.size(); i-- > 0;) {
            new_strides[i] = stride;
            stride *= static_cast<ptrdiff_t>(new_shape[i]);
        }
        return true;
    }

    struct Chunk {
        size_t size;
        ptrdiff_t stride;
    };

    std::vector<Chunk> chunks;
    size_t idx = compact_shape.size();
    while (idx > 0) {
        size_t end = idx - 1;
        size_t chunk_size = compact_shape[end];
        ptrdiff_t chunk_stride = compact_strides[end];
        size_t inner_size = compact_shape[end];
        ptrdiff_t inner_stride = compact_strides[end];
        while (end > 0) {
            size_t prev = end - 1;
            if (compact_strides[prev] == inner_stride * static_cast<ptrdiff_t>(inner_size)) {
                chunk_size *= compact_shape[prev];
                inner_size = compact_shape[prev];
                inner_stride = compact_strides[prev];
                end = prev;
            } else {
                break;
            }
        }
        chunks.push_back({chunk_size, chunk_stride});
        idx = end;
    }

    size_t new_i = new_shape.size();
    for (size_t c = 0; c < chunks.size(); c++) {
        size_t chunk_size = chunks[c].size;
        ptrdiff_t base_stride = chunks[c].stride;
        size_t prod = 1;
        while (new_i > 0 && prod < chunk_size) {
            size_t dim = new_i - 1;
            size_t size = new_shape[dim];
            new_strides[dim] = base_stride * static_cast<ptrdiff_t>(prod);
            if (size != 1) {
                prod *= size;
            }
            new_i--;
        }
        if (prod != chunk_size) {
            return false;
        }
    }

    while (new_i > 0) {
        size_t dim = new_i - 1;
        if (new_shape[dim] != 1) {
            return false;
        }
        ptrdiff_t stride = (dim + 1 < new_shape.size()) ? new_strides[dim + 1] : 1;
        new_strides[dim] = stride;
        new_i--;
    }

    return true;
}
}

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    if (_meta.shape.empty()) {
        return true;
    }
    if (this->numel() == 0) {
        return true;
    }
    size_t expected = 1;
    for (size_t i = _meta.shape.size(); i-- > 0;) {
        if (_meta.shape[i] != 1 && _meta.strides[i] != static_cast<ptrdiff_t>(expected)) {
            return false;
        }
        expected *= _meta.shape[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(order.size() == this->ndim(), "invalid order");
    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());
    std::vector<bool> seen(order.size(), false);
    for (size_t i = 0; i < order.size(); i++) {
        size_t dim = order[i];
        CHECK_ARGUMENT(dim < this->ndim(), "invalid order");
        CHECK_ARGUMENT(!seen[dim], "invalid order");
        seen[dim] = true;
        new_shape[i] = _meta.shape[dim];
        new_strides[i] = _meta.strides[dim];
    }
    TensorMeta meta{_meta.dtype, std::move(new_shape), std::move(new_strides)};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = numel_from_shape(shape);
    CHECK_ARGUMENT(new_numel == this->numel(), "invalid shape");
    std::vector<ptrdiff_t> new_strides;
    CHECK_ARGUMENT(
        compute_view_strides(this->shape(), this->strides(), shape, new_strides),
        "incompatible view");
    TensorMeta meta{_meta.dtype, shape, std::move(new_strides)};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < this->ndim(), "invalid dim");
    CHECK_ARGUMENT(start <= end, "invalid slice range");
    CHECK_ARGUMENT(end <= _meta.shape[dim], "invalid slice range");
    TensorMeta meta = _meta;
    meta.shape[dim] = end - start;
    size_t new_offset = _offset + start * static_cast<size_t>(_meta.strides[dim]) * this->elementSize();
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());
    size_t bytes = this->numel() * this->elementSize();
    if (_storage->isHost()) {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            bytes,
            LLAISYS_MEMCPY_H2H);
    } else {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            bytes,
            LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
