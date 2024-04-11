#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <sstream>
#include <sys/types.h>
#include <tuple>
#include <vector>

#include "include/Utils.h"
#include "CImg.h"

#include <CL/opencl.hpp>

#include "include/dtypes.h"

using namespace cimg_library;

template <typename T>
auto str_vec(std::vector<T> vector) -> std::string {
	std::ostringstream oss;
	size_t counter = 0;
	const size_t size = vector.size();

	oss << "{";	
	for (const T& val: vector)
		oss << val << ((counter++ < size - 1)? ", " : "");
	oss << "}";

	return oss.str();
}

void print_platform(ci32& platform_id, ci32& device_id) {
	std::cout
		<< "Running on "
		<< GetPlatformName(platform_id)
		<< ", "
		<< GetDeviceName(platform_id, device_id);
}

void print_help_message() {
	std::cout
		<< "-h = print this help message\n"
		<< "-p = print platform+device id\n"
		<< "-d = print debug messages\n"
		<< "-c <gs|rgb> = specifies whether to interpret the image as greyscale or color (defaults to greyscale)\n"
		<< "-s <8|16> = specifies the color rate of the image (defaults to 8)\n"
		<< "-i <filename> = specifies the input file to use\n";
}

enum ColorMode {GRAYSCALE, RGB};

struct Options {
	bool debug, help_mode;
	size_t bits;
	ColorMode color_mode;
	std::string file_name;

	Options(): help_mode(true) {}
	Options(bool debug, bool help_mode, size_t bits, ColorMode color_mode, std::string file_name):
		debug(debug), help_mode(help_mode), bits(bits), color_mode(color_mode), file_name(file_name) {}
};

auto handle_args(ci32& argc, str* argv, ci32& platform_id, ci32& device_id) -> Options {
	bool debug = false;
	size_t bits = 8;
	ColorMode color_mode = GRAYSCALE;
	std::string file_name = "";
	
	for (i32 i = 0; i < argc; ++i) {
		const std::string str_arg(argv[i]);
		const std::string next_arg((i < argc - 1)? argv[i + 1] : "");
		
		if (str_arg == "-h") {
			print_help_message();
			return Options();
		}
		
		if (str_arg == "-p") print_platform(platform_id, device_id);
		if (str_arg == "-d") debug = True;
		
		if (str_arg == "-c") {
			if (next_arg == "gs") {}
			else if (next_arg == "rgb") color_mode = RGB;
			else throw std::invalid_argument("-c option must be either rgb or gs");
		}
		if (str_arg == "-s") {
			if (next_arg == "8") {}
			else if (next_arg == "16") bits = 16;
			else throw std::invalid_argument("-s option must be either 8 or 16");
		}
		if (str_arg == "-i") {
			file_name = next_arg;
		}
	}

	if (file_name.empty()) throw std::invalid_argument("a file name must be specified with -i <filename>");
	
	return Options(debug, false, bits, color_mode, file_name);
}

void print_build_status(const cl::Program& program, const cl::Context& context) {
	auto context_info = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	auto build_status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context_info);
	auto build_options = program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context_info);
	auto build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context_info);
	std::cout
		<< "Build Status:\n"
		<< build_status
		<< "\nBuild Options:\n"
		<< build_options
		<< "\nBuild Log:\n"
		<< build_log
		<< '\n';
}

template<typename T>
void print_image_info(const CImg<T> c_img) {
	std::cout
		<< "width: "
		<< c_img.width()
		<< ", height: "
		<< c_img.height()
		<< ", depth: "
		<< c_img.depth()
		<< ", spectrum: "
		<< c_img.spectrum()
		<< "\n";
}

std::string relative_path() {
	const std::string path(__FILE__);
	const std::string main_f("main.cpp");
	const auto index = path.find(main_f);
	return path.substr(0, index);
}

template <typename T>
class HistFilter {
	std::string _image_filename, _kernel_filename;
	i32         _platform_id, _device_id;
	ColorMode   _color_mode;
	bool        _debug;
	
	cl::Context          _context;
	cl::CommandQueue     _queue;
	cl::Program::Sources _sources;
	cl::Program          _program;
	cl::Device           _device;
	
	auto _max_int() -> size_t;
	auto _type_prefix() -> std::string;
	auto _load_image(const std::string&) -> std::pair<CImg<T>, CImgDisplay>;

public:
	HistFilter(HistFilter<T>&) = delete;

	HistFilter(
		const std::string& image_filename,
		const std::string& kernel_filename,
		ci32& platform_id,
		ci32& device_id,
		const ColorMode& color_mode,
		cbool& debug
	):
		_image_filename(image_filename),
		_kernel_filename(kernel_filename),
		_platform_id(platform_id),
		_device_id(device_id),
		_color_mode(color_mode),
		_debug(debug)
	{
		/*
		A cl::Context is used so that opencl can manage memory, devives and error handling.
		Then a cl::CommandQueue is created so that opencl commands can be queued and ran asynchronously.
		A cl::ProgramSources class is used to retrieve the opencl source code from kernels.cl and then
		the program is constructed using both our context and sources.
		*/
		_context = GetContext(_platform_id, _device_id);
		_queue = cl::CommandQueue(_context);
		AddSources(_sources, _kernel_filename);
		_program = cl::Program(_context, _sources);
		_device = _context.getInfo<CL_CONTEXT_DEVICES>().front();

		// program is built. if debug is enabled the build status is printed regardless of failure.
		try {
			_program.build();
			if (debug) print_build_status(_program, _context);
		}
		catch (const cl::Error& err) {
			if (!debug) print_build_status(_program, _context);
			throw err;
		}
	}

	void output();
};

template<typename T>
auto HistFilter<T>::_max_int() -> size_t {
	return 2 << (sizeof(T) * 8 - 1);
}

template<typename T>
auto HistFilter<T>::_type_prefix() -> std::string {
	switch (sizeof(T)) {
		case 1: return std::string("uchar_");
		case 2: return std::string("ushort_");
		// case 4: return std::string("uint_");
	}
	return std::string("not_found_");
}

template<typename T>
auto HistFilter<T>::_load_image(const std::string& image_filename) -> std::pair<CImg<T>, CImgDisplay> {
	/*
	This sections loads the input image into a cimage_library::CImg<T>
	and and passes it by reference into a cimage_library::CImgDisplay so that it can later be displayed
	*/
	CImg<T> image_input(image_filename.c_str());
	
	return std::pair<CImg<T>, CImgDisplay>(
		image_input,
		CImgDisplay(image_input, "input")
	);
}

template<typename T>
void HistFilter<T>::output() {
	//detect any potential exceptions
	// this would look better with c++17 structured bindings but alas

	auto prefix = _type_prefix();
	
	auto input = _load_image(_image_filename);
	auto& input_image = input.first;
	auto& input_disp = input.second;
	const auto input_size = (size_t)input_image.size();
	const auto input_height = input_image.height();
	const auto input_width = input_image.width();
	const auto input_depth = input_image.depth();
	const auto input_spectrum = input_image.spectrum();
	const auto input_pixels = (_color_mode == RGB)? input_size / 3 : input_size;
	
	std::cout << "checkpoint 1\n";
	
	std::cout << "input_size   " << input_size << "\n";
	std::cout << "input_pixels " << input_pixels << "\n";
	
	// loading input data into buffer
	cl::Buffer input_buffer(_context, CL_MEM_READ_ONLY, input_size * sizeof(T));
	_queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_size * sizeof(T), &input_image.data()[0]);

	std::cout << "checkpoint 2\n";

	cl::Kernel kernel;
	cl::Buffer cmyk_buffer, k_buffer;
	std::vector<T> cmyk_vector(4 * input_pixels);

	// if the image is RGB then we need to convert to hsl because it has a lightness channel.
	if (_color_mode == RGB) {
		// creating new hsl image buffer
		cmyk_buffer = cl::Buffer(_context, CL_MEM_READ_WRITE, 4 * input_pixels * sizeof(T));

		// specifying kernel, passing kernel arguments and then loading it into queue
		kernel = cl::Kernel(_program, (prefix + "rgb_to_cmyk").c_str());
		kernel.setArg(0, input_buffer);
		kernel.setArg(1, cmyk_buffer);
	
		_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_size), cl::NullRange);
	
		// retrieving the hsl data
		_queue.enqueueReadBuffer(cmyk_buffer, CL_TRUE, 0, 4 * input_pixels * sizeof(T), &cmyk_vector.data()[0]);

		// only the lightness part of the cmyk array is used to make the histogram so the corresponding data slice is copied
		k_buffer = cl::Buffer(_context, CL_MEM_READ_WRITE, input_pixels * sizeof(T));
		_queue.enqueueCopyBuffer(cmyk_buffer, k_buffer, 3 * input_pixels * sizeof(T), 0, input_pixels * sizeof(T));
	}
	
	// hist buffer must have a large int type to prevent overflowing. If the image was all one color
	// for example, it would be a problem because one value of the histogram would get overflowed.
	const size_t hist_items = _max_int();
	const size_t hist_size = hist_items * sizeof(u32);
	std::vector<u32> hist_vector(hist_items);
	cl::Buffer hist_buffer(_context, CL_MEM_READ_WRITE, hist_size);

	std::cout << "checkpoint 3\n";

	// histogram is then produced using hist kernel
	kernel = cl::Kernel(_program, (prefix + "hist").c_str());
	kernel.setArg(0, (_color_mode == RGB)? k_buffer : input_buffer);
	kernel.setArg(1, hist_buffer);
	kernel.setArg(2, cl::Local(2048 * sizeof(u32)));
	kernel.setArg(3, hist_items);
	
	_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_pixels), cl::NullRange);

	std::cout << "checkpoint 4\n";
	
	_queue.enqueueReadBuffer(hist_buffer, CL_TRUE, 0, hist_size, &hist_vector.data()[0]);

	std::cout << "checkpoint 5\n";

	if (_debug) std::cout << "Histogram:\n" << str_vec(hist_vector) << "\n";

	// a normalized cdf is then produced from the histogram
	std::vector<T> cdf_vector(hist_items);
	cl::Buffer cdf_buffer(_context, CL_MEM_READ_WRITE, hist_items * sizeof(T));

	std::cout << "checkpoint 6\n";
	
	kernel = cl::Kernel(_program, (prefix + "cdf").c_str());
	kernel.setArg(0, hist_buffer);
	kernel.setArg(1, cdf_buffer);
	kernel.setArg(2, input_pixels);
	kernel.setArg(3, hist_items);
	
	_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(hist_items), cl::NullRange);

	std::cout << "checkpoint 7\n";
		
	_queue.enqueueReadBuffer(cdf_buffer, CL_TRUE, 0, hist_items * sizeof(T), &cdf_vector.data()[0]);

	std::cout << "checkpoint 8\n";

	if (_debug) std::cout << "Normalised CDF:\n" << str_vec(cdf_vector) << "\n";

	std::vector<T> output_vector(input_size);

	if (_color_mode == RGB) {
		// cdf is then used to create an equalized version of the lightness buffer
		cl::Buffer equalized_k_buffer(_context, CL_MEM_READ_WRITE, input_pixels * sizeof(T));
		_queue.enqueueCopyBuffer(cmyk_buffer, equalized_k_buffer, 3 * input_pixels * sizeof(T), 0, input_pixels);
	
		std::cout << "checkpoint 9 A\n";
		
		kernel = cl::Kernel(_program, (prefix + "cdf_lookup").c_str());
		kernel.setArg(0, equalized_k_buffer);
		kernel.setArg(1, cdf_buffer);

		_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_pixels), cl::NullRange);

		std::cout << "checkpoint 13 A\n";

		_queue.enqueueCopyBuffer(equalized_k_buffer, cmyk_buffer, 0, 3 * input_pixels * sizeof(T), input_pixels * sizeof(T));
			
		// the new equalized 
		cl::Buffer output_buffer(_context, CL_MEM_READ_WRITE, input_size * sizeof(T));
		kernel = cl::Kernel(_program, (prefix + "cmyk_to_rgb").c_str());
		kernel.setArg(0, /*equalized_*/cmyk_buffer);
		kernel.setArg(1, output_buffer);

		_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(4 * input_pixels), cl::NullRange);

		std::cout << "checkpoint 14 A\n";
		
		_queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, input_size * sizeof(T), &output_vector.data()[0]);

		std::cout << "checkpoint 15 A\n";
	}
	else {
		
		cl::Buffer output_buffer(_context, CL_MEM_READ_ONLY, input_size * sizeof(T));
		_queue.enqueueWriteBuffer(output_buffer, CL_TRUE, 0, input_size * sizeof(T), &input_image.data()[0]);

		kernel = cl::Kernel(_program, (prefix + "cdf_lookup").c_str());
		kernel.setArg(0, output_buffer);
		kernel.setArg(1, cdf_buffer);

		std::cout << "checkpoint 9 B\n";
		
		_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_pixels), cl::NullRange);

		std::cout << "checkpoint 10 B\n";
		
		_queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, input_size * sizeof(T), &output_vector.data()[0]);
	}
	
	// displaying hsl image
	CImg<T> output_image(output_vector.data(), input_width, input_height, input_depth, input_spectrum);
	CImgDisplay output_disp(output_image, "output");

	while (!output_disp.is_keyESC() && !output_disp.is_closed()) output_disp.wait(1);
}

auto main(i32 argc, str* argv) -> i32 {
	// sets up some constants later used in the cl::Context and the relatative path of the files to be used
	ci32 platform_id = 0;
	ci32 device_id   = 0;
	const std::string path = relative_path();
	// const std::string image_filename  = path + "images/test.ppm";
	const std::string kernel_filename = path + "kernels/kernels.cl";

	cimg::exception_mode(0);

	try {
		auto options = handle_args(argc, argv, platform_id, device_id);
		if (options.help_mode) return EXIT_SUCCESS;
		
		switch (options.bits) {
			case 8: {
				HistFilter<u8> hist_filter(
					path + "images/" + options.file_name,
					kernel_filename,
					platform_id,
					device_id,
					options.color_mode,
					options.debug
				);
				hist_filter.output();
				break;
			}
			case 16: {
				HistFilter<u16> hist_filter(
					path + "images/" + options.file_name,
					kernel_filename,
					platform_id,
					device_id,
					options.color_mode,
					options.debug
				);
				hist_filter.output();
				break;
			}
		}
	}
	catch (const std::invalid_argument& err) {
		std::cerr << "Argument Error: " << err.what() << "\nfor help on option try -h" << std::endl;
		return EXIT_FAILURE;
	}
	catch (const cl::Error& err) {
		std::cerr << "OpenCL Error: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
		return EXIT_FAILURE;
	}
	catch (const CImgException& err) {
		std::cerr << "CImg Error: " << err.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
