#include <array>
#include <iostream>
#include <vector>
#include <string>

#include <CL/opencl.hpp>

#include "Utils.h"
#include "CImg.h"
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
	std::vector<str> argv_vec(argv, argv + sizeof(str) * argc);
	for (cstr arg : argv_vec) {
		std::string str_arg(arg);
		if (str_arg == "-p")
			print_platform(platform_id, device_id);
		if (str_arg == "-d")
			debug = true;
	}
}

auto main(i32 argc, str* argv) -> i32 {
	ci32 platform_id = 0;
	ci32 device_id = 0;
	bool debug = false;
	handle_args(argc, argv, platform_id, device_id, debug);

	cstr image_filename = "test.ppm";
	cstr kernel_fileneame = "kernels.cl";
	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<u8> image_input(image_filename);
		CImgDisplay disp_input(image_input, "input");
		auto context = GetContext(platform_id, device_id);

		cl::CommandQueue queue(context);
		cl::Program::Sources sources;
		AddSources(sources, kernel_fileneame);
		cl::Program program(context, sources);

		build_kernel(program, context, debug);

		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		cl::Kernel kernel = cl::Kernel(program, "identity");
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_image_output);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);

		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
