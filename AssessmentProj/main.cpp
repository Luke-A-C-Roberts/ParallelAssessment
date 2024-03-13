#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>

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

void build_kernel(const cl::Program& program, const cl::Context& context, cbool debug = false) {
	try {
		program.build();
		if (debug) print_build_status(program, context);
	}
	catch (const cl::Error& err) {
		if (!debug) print_build_status(program, context);
		throw err;
	}
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

template<typename T>
auto load_image(const std::string& image_filename) -> std::pair<CImg<T>, CImgDisplay> {
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

std::string relative_path() {
	const std::string path(__FILE__);
	const std::string main_f("main.cpp");
	const auto index = path.find(main_f);
	return path.substr(0, index);
}

template <typename T>
auto make_io_buffers(
	const cl::Context& context,
	const UINT_PTR& size
) -> std::pair<cl::Buffer, cl::Buffer> {
	/*
	Creates two cl::Buffer structs for input and output. Input only has read permissions while output can
	be read and written.
	*/
	return std::pair<cl::Buffer, cl::Buffer>(
		cl::Buffer(context, CL_MEM_READ_ONLY, size),
		cl::Buffer(context, CL_MEM_READ_WRITE, size) //should be the same as input image
	);
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

	std::cout << __FILE__ << "\n";

	//detect any potential exceptions
	try {
		// this would look better with c++17 structured bindings but alas
		// Image
		auto input = load_image<u8>(image_filename);
		auto& image_input = input.first;
		auto& disp_input = input.second;
		auto size_input = image_input.size();

		/*
		A cl::Context is used so that opencl can manage memory, devives and error handling.
		Opencl errors are passed back to the program and handeled by catch in with a cl::Error.
		Then a cl::CommandQueue is created so that opencl commands can be queued and ran asynchronously.
		A cl::ProgramSources class is used to retrieve the opencl source code from kernels.cl and then
		the program is constructed using both our context and sources. Finally `build_kernel` (see above)
		is called to build `program` and handel debuging output and exceptions.
		*/
		auto context = GetContext(platform_id, device_id);
		cl::CommandQueue queue(context);
		cl::Program::Sources sources;
		AddSources(sources, kernel_filename);
		cl::Program program(context, sources);
		build_kernel(program, context, debug);

		// cl::Buffer created to hold image data
		auto io_buffers = make_io_buffers<u8>(context, size_input);
		auto& dev_image_input = io_buffers.first;
		auto& dev_image_output = io_buffers.second;

		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		cl::Kernel kernel = cl::Kernel(program, "identity");
		 kernel.setArg(0, dev_image_input);
		 kernel.setArg(1, dev_image_output);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);

		vector<u8> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<u8> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");
		
		while (!disp_input.is_closed() && !disp_output.is_closed() && !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (const CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
