#pragma once
#include <map>
#include <string>
#include <iostream>
#include <stdexcept>

// CRTP base for backend-specific type wrappers.
//
// Derived classes must provide:
//   static const std::map<KeyType, BackendEnum>& get_mappings();
//
// Template parameters:
//   Derived     - the concrete backend type (CRTP)
//   BaseType    - the generic mblas type (mblas_operation, mblas_data_type, mblas_compute_type)
//   BackendEnum - the vendor-specific enum type (cublasOperation_t, cudaDataType, etc.)
//   KeyType     - the map key type (defaults to BaseType; compute types use mblas_compute_type_enum)
template <typename Derived, typename BaseType, typename BackendEnum, typename KeyType = BaseType>
class mblas_backend_type : public BaseType {
public:
    // Forward all constructors from the base type
    using BaseType::BaseType;

    // Copy assignment
    Derived& operator=(const Derived& other) {
        if (this == &other)
            return static_cast<Derived&>(*this);
        this->set(other);
        return static_cast<Derived&>(*this);
    }

    // Backend conversion: looks up the vendor enum from the static map
    BackendEnum convert() const {
        const auto& mappings = Derived::get_mappings();
        auto it = mappings.find(static_cast<KeyType>(*this));
        if (it != mappings.end()) {
            return it->second;
        }
        std::cout << "Failed to convert type: " << this->to_string() << std::endl;
        throw std::out_of_range("Unsupported type conversion");
    }

    // Implicit conversion to vendor enum
    operator BackendEnum() const { return convert(); }
};
