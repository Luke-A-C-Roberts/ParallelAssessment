//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id   = get_global_id(0);
	int size = get_global_size(0);

	if (id % (size / 3) == 0)
		printf("%d: %d\n", id, A[id]);

	B[id] = A[id];
}


kernel void filter_h(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//this is just a copy operation, modify to filter out the individual colour channels
	if (colour_channel == 0) {
		B[id] = 100;
	}
	else {
		B[id] = A[id];
	}
}

kernel void u8_to_hsl_f32(global const uchar* input, global float* output) {
	int width = get_global_size(0);
	int height = get_global_size(1);
	int image_size = width * height;
	int channels = get_global_size(2);
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x >= width || y >= height)
		return;

	int id = y * width * 3 + x * 3;
	float r = input[id] / 255.0;
	float g = input[id + 1] / 255.0;
	float b = input[id + 2] / 255.0;

	// https://www.niwa.nu/2013/05/math-behind-colorspace-conversions-rgb-hsl/
	float cmax = fmax(fmax(r, g), b);
	float cmin = fmin(fmin(r, g), b);
	float cdelta = cmax - cmin;
	float csum = cmax + cmin;

	float l = csum / 2.0;
	float s = (l <= 0.5) ? (cdelta / csum) : (cdelta / (2.0 - cmax - cmin));
	float h;
	if (cmax == r) {
		h = (g - b) / cdelta;
	}
	else if (cmax == g) {
		h = 2.0 + (b - r) / cdelta;
	}
	else if (cmax == b) {
		h = 4.0 + (r - g) / cdelta;
	}


	output[id] = h;
	output[id + 1] = s;
	output[id + 2] = l;
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	uint result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size];

		result /= 9;
	}

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];
	}

	B[id] = (uchar)result;
}