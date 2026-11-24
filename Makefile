CXX ?= clang++
CLANG_FORMAT ?= clang-format
CLANG_TIDY ?= clang-tidy
JOBS ?= 8

SRCS := src/main.cpp src/verify.cpp src/verify.hpp src/matrix-multiply.cpp src/matrix-multiply.hpp \
	src/benchmark.cpp src/opencv_benchmark.cpp
TIDY_SRCS := src/main.cpp src/verify.cpp src/matrix-multiply.cpp src/benchmark.cpp

.PHONY: all asan tsan format format-check tidy clean

all:
	cmake -S . -B build -DCMAKE_CXX_COMPILER=$(CXX) -DCMAKE_CXX_FLAGS="-O2 -Wall -Wextra"
	cmake --build build --target matmul-bs64 -j $(JOBS)

asan:
	cmake -S . -B build-asan -DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CXX_FLAGS="-g -O2 -fsanitize=address,undefined -fno-omit-frame-pointer"
	cmake --build build-asan --target matmul-bs64 -j $(JOBS)
	./build-asan/matmul-bs64 --benchmark_filter="neon_4x4_bs64/256|blocked_bs64/256" --benchmark_min_time=0.01s

tsan:
	cmake -S . -B build-tsan -DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CXX_FLAGS="-g -O2 -fsanitize=thread"
	cmake --build build-tsan --target matmul-bs64 -j $(JOBS)
	./build-tsan/matmul-bs64 --benchmark_filter="neon_4x4_bs64/256|blocked_bs64/256" --benchmark_min_time=0.01s

format:
	@command -v $(CLANG_FORMAT) >/dev/null 2>&1 || { echo "clang-format not found (install via: brew install clang-format)"; exit 1; }
	$(CLANG_FORMAT) -i $(SRCS)

format-check:
	@command -v $(CLANG_FORMAT) >/dev/null 2>&1 || { echo "clang-format not found (install via: brew install clang-format)"; exit 1; }
	$(CLANG_FORMAT) --dry-run --Werror $(SRCS)

compile_commands.json:
	cmake -S . -B build -DCMAKE_CXX_COMPILER=$(CXX) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	cp build/compile_commands.json compile_commands.json

tidy: compile_commands.json
	@command -v $(CLANG_TIDY) >/dev/null 2>&1 || { echo "clang-tidy not found (install via: brew install llvm)"; exit 1; }
	$(CLANG_TIDY) -p . $(TIDY_SRCS)

clean:
	rm -rf build-asan build-tsan compile_commands.json
