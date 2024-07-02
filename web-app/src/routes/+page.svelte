<script lang="ts">
	import { onMount } from 'svelte';

	let canvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D;
	let prediction: string | null = null; // Holds the prediction result
	let error: string | null = null; // Holds any error message

	onMount(() => {
		setupCanvas();
	});

	function setupCanvas() {
		ctx = canvas.getContext('2d')!;
		canvas.width = 200;
		canvas.height = 200;
		ctx.fillStyle = 'white';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
	}

	function clearCanvas() {
		ctx.fillStyle = 'white';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		prediction = null; // Clear the prediction when the canvas is cleared
		error = null; // Clear any error message
	}

	function getImageData(): string {
		return canvas.toDataURL('image/png');
	}

	async function sendImage() {
		try {
			const imageData = getImageData();
			const base64Data = imageData.split(',')[1];

			const response = await fetch('http://localhost:5000/predict', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ image: base64Data }) // Split and send only the Base64 part
			});
			if (!response.ok) {
				throw new Error('Network response was not ok');
			}
			const result = await response.json();
			prediction = `Predicted Number: ${result.prediction}`;
			error = null;
		} catch (err) {
			error = 'Failed to predict the number. Please try again.';
			console.error(err);
		}
	}

	let drawing = false;

	function startDrawing(e: MouseEvent) {
		drawing = true;
		ctx.beginPath();
	}

	function stopDrawing() {
		drawing = false;
		ctx.beginPath();
	}

	function draw(event: MouseEvent) {
		if (!drawing) return;

		// Calculate scale factors
		const scaleX = canvas.width / canvas.offsetWidth;
		const scaleY = canvas.height / canvas.offsetHeight;

		// Adjust mouse coordinates
		const mouseX = (event.clientX - canvas.offsetLeft) * scaleX;
		const mouseY = (event.clientY - canvas.offsetTop) * scaleY;

		ctx.lineWidth = 16;
		ctx.lineCap = 'round';
		ctx.lineTo(mouseX, mouseY);
		ctx.stroke();
	}
</script>

<div class="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
	<div class="shadow-lg p-6 rounded-lg bg-white">
        <h2 class="text-center text-2xl font-bold mb-4">Number Prediction</h2>
        
		<canvas
			class="border border-gray-300 cursor-crosshair w-full max-w-lg h-64"
			bind:this={canvas}
			on:mousedown={startDrawing}
			on:mouseup={stopDrawing}
			on:mouseleave={stopDrawing}
			on:mousemove={draw}
		></canvas>

		<div class="flex justify-center space-x-4 mt-4">
			<button
				on:click={sendImage}
				class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded transition duration-150 ease-in-out"
				>Predict Number</button
			>
			<button
				on:click={clearCanvas}
				class="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-6 rounded transition duration-150 ease-in-out"
				>Clear</button
			>
		</div>

		{#if prediction}
			<p class="mt-4 text-center text-green-600">{prediction}</p>
		{:else}
			<p class="mt-4 text-center text-gray-500">Please draw a number...</p>
		{/if}

		{#if error}
			<p class="mt-4 text-center text-red-600">{error}</p>
		{/if}
	</div>
</div>
