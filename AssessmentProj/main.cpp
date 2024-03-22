#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <cstdlib>

#include "Utils.h"
#include "CImg.h"

#include <CL/opencl.hpp>

#include "dtypes.h"

using namespace cimg_library;

void print_platform(ci32& platform_id, ci32& device_id) {
	std::cout
		<< "Running on "
		<< GetPlatformName(platform_id)
		<< ", "
		<< GetDeviceName(platform_id, device_id)
		<< '\n';
}

void handle_args(ci32& argc, str* argv, ci32& platform_id, ci32& device_id, bool& debug) {
	for (i32 i = 0; i < argc; ++i) {
		std::string str_arg(argv[i]);
		if (str_arg == "-p")
			print_platform(platform_id, device_id);
		if (str_arg == "-d")
			debug = true;
	}
}

void print_build_status(const cl::Program& program, const cl::Context& context) {
	auto build_status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]);
	auto build_options = program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]);
	auto build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]);
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
	i32 _platform_id, _device_id;
	bool _debug;

	void _build_kernel(const cl::Program&, const cl::Context&, cbool&);
	auto _load_image(const std::string&, const std::string&) -> std::pair<CImg<T>, CImgDisplay>;

public:
	HistFilter(
		const std::string& image_filename,
		const std::string& kernel_filename,
		ci32& platform_id,
		ci32& device_id,
		cbool& debug
	): _image_filename(image_filename),
	   _kernel_filename(kernel_filename),
	   _platform_id(platform_id),
	   _device_id(device_id),
	   _debug(debug) {}

	void output();
};

template <typename T>
void HistFilter<T>::_build_kernel(const cl::Program& program, const cl::Context& context, cbool& debug) {
	try {
		program.build();
		if (debug) print_build_status(program, context);
	}
	catch (const cl::Error& err) {
		if (!debug) print_build_status(program, context);
		throw err;
	}
}

template<typename T>
auto HistFilter<T>::_load_image(const std::string& image_filename, const std::string& mode) -> std::pair<CImg<T>, CImgDisplay> {
	/*
	This sections loads the input image into a cimage_library::CImg<T>
	and and passes it by reference into a cimage_library::CImgDisplay so that it can later be displayed
	*/
	CImg<T> image_input(image_filename.c_str());
	if (mode == "rgb") { }
	else if (mode == "hsl") {
		image_input = image_input.RGBtoHSL();
	}
	else if (mode == "hsv") {
		image_input = image_input.RGBtoHSV();
	}
	
	return std::pair<CImg<T>, CImgDisplay>(
		image_input,
		CImgDisplay(image_input, "input")
	);
}

class BufferMapper {
	cl::Program& _program;
	cl::CommandQueue& _queue;
	cl::Buffer& _input_buffer;
	cl::Buffer& _output_buffer;
	UINT_PTR _size;

public:
	BufferMapper(
		cl::Program& program,
		cl::CommandQueue& queue,
		cl::Buffer& input_buffer,
		cl::Buffer& output_buffer,
		const UINT_PTR& size
	):  _program(program),
		_queue(queue),
	    _input_buffer(input_buffer),
		_output_buffer(output_buffer),
		_size(size) {}

	void map(const std::string&);
	void map(const std::string&, cu32&, cu32&);
};

void BufferMapper::map(const std::string& kernel_function_name) {

	cl::Kernel kernel = cl::Kernel(_program, kernel_function_name.c_str());
	kernel.setArg(0, _input_buffer);
	kernel.setArg(1, _output_buffer);

	_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(_size), cl::NullRange);
}

void BufferMapper::map(const std::string& kernel_function_name, cu32& width, cu32& height) {

	cl::Kernel kernel = cl::Kernel(_program, kernel_function_name.c_str());
	kernel.setArg(0, _input_buffer);
	kernel.setArg(1, _output_buffer);
	kernel.setArg(2, width);
	kernel.setArg(3, height);

	_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(_size), cl::NullRange);
}

template<typename T>
void HistFilter<T>::output() {
	//detect any potential exceptions
	// this would look better with c++17 structured bindings but alas
	// Image
	auto input = _load_image(_image_filename, "rgb");
	auto& input_image = input.first;
	auto& input_disp = input.second;
	const auto input_size = input_image.size();
	const auto input_height = input_image.height();
	const auto input_width = input_image.width();
	const auto input_depth = input_image.depth();
	const auto input_spectrum = input_image.spectrum();

	/*
	A cl::Context is used so that opencl can manage memory, devives and error handling.
	Opencl errors are passed back to the program and handeled by catch in with a cl::Error.
	Then a cl::CommandQueue is created so that opencl commands can be queued and ran asynchronously.
	A cl::ProgramSources class is used to retrieve the opencl source code from kernels.cl and then
	the program is constructed using both our context and sources. Finally `build_kernel` (see above)
	is called to build `program` and handel debuging output and exceptions.
	*/
	auto context = GetContext(_platform_id, _device_id);
	cl::CommandQueue queue(context);
	cl::Program::Sources sources;
	AddSources(sources, _kernel_filename);
	cl::Program program(context, sources);
	_build_kernel(program, context, _debug);

	cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, input_size);
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_size, &input_image.data()[0]);

	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, input_size);
	BufferMapper(program, queue, input_buffer, output_buffer, input_size).map("u8hsl");

	std::vector<T> output_vector(input_size);
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, input_size, &output_vector.data()[0]);
	CImg<T> output_image(output_vector.data(), input_width, input_height, input_depth, input_spectrum);
	CImgDisplay output_disp(output_image, "output");


	while (!output_disp.is_closed() && !output_disp.is_keyESC()) {
		output_disp.wait(1);
	}
}

auto main(i32 argc, str* argv) -> i32 {
	// sets up some constants later used in the cl::Context and the relatative path of the files to be used
	ci32 platform_id = 0;
	ci32 device_id = 0;
	std::string path = relative_path();
	std::string image_filename = path + "images\\test.ppm";
	std::string kernel_filename = path + "kernels\\kernels.cl";

	// debug used for testing and activated in handle_args if -d is in argv
	bool debug = false;
	handle_args(argc, argv, platform_id, device_id, debug);

	cimg::exception_mode(0);

	try {
		HistFilter<u8> hist_filter(image_filename, kernel_filename, platform_id, device_id, debug);
		hist_filter.output();
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (const CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return EXIT_SUCCESS;
}

