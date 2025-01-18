document.addEventListener('DOMContentLoaded', async () => {
    // DOM Elements
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadContainer = document.getElementById('upload-container');
    const previewContainer = document.getElementById('preview-container');
    const originalImg = document.getElementById('original-img');
    const processedImg = document.getElementById('processed-img');
    const downloadBtn = document.getElementById('download-btn');
    const tryAgainBtn = document.getElementById('try-again-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const uploadProgress = document.getElementById('upload-progress');
    const progressBar = uploadProgress.querySelector('.progress-bar');
    const aiPrecision = document.getElementById('ai-precision');
    const outputFormat = document.getElementById('output-format');

    let bodyPixNet = null;

    // Initialize BodyPix model
    async function initializeAI() {
        try {
            bodyPixNet = await bodyPix.load({
                architecture: 'ResNet50',
                outputStride: 16,
                quantBytes: 4
            });
        } catch (error) {
            console.error('Error loading AI model:', error);
        }
    }

    // Initialize AI on page load
    initializeAI();

    // Theme Toggle
    themeToggle.addEventListener('click', () => {
        body.classList.toggle('dark-mode');
        const icon = themeToggle.querySelector('i');
        if (body.classList.contains('dark-mode')) {
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        } else {
            icon.classList.remove('fa-sun');
            icon.classList.add('fa-moon');
        }
    });

    // Drag and Drop Functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('highlight');
    }

    function unhighlight() {
        dropZone.classList.remove('highlight');
    }

    // Handle File Drop and Upload
    dropZone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);
    dropZone.addEventListener('click', () => fileInput.click());

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                processImage(file);
            } else {
                showError('Please upload an image file.');
            }
        }
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        uploadContainer.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 3000);
    }

    // Image Processing using multiple AI models
    async function processImage(file) {
        try {
            // Show loading state
            loadingOverlay.classList.add('active');
            uploadProgress.classList.add('active');
            uploadContainer.style.display = 'none';
            previewContainer.style.display = 'block';

            // Load and display original image
            const imageUrl = URL.createObjectURL(file);
            originalImg.src = imageUrl;

            // Update progress
            updateProgress(20);

            // Create image element for processing
            const img = document.createElement('img');
            img.src = imageUrl;
            await img.decode();

            // Update progress
            updateProgress(40);

            // Get AI precision settings
            const precision = aiPrecision.value;
            const segmentationConfig = getSegmentationConfig(precision);

            // Make segmentation
            const segmentation = await bodyPixNet.segmentPerson(img, segmentationConfig);

            // Update progress
            updateProgress(60);

            // Process the segmentation
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');

            // Draw original image
            ctx.drawImage(img, 0, 0);

            // Update progress
            updateProgress(80);

            // Get image data
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const pixels = imageData.data;

            // Apply advanced processing based on precision
            await enhanceSegmentation(pixels, segmentation.data, precision);

            // Put processed image data back
            ctx.putImageData(imageData, 0, 0);

            // Update progress
            updateProgress(90);

            // Convert to desired format
            const format = outputFormat.value;
            const outputQuality = format === 'jpeg' ? 0.9 : undefined;
            const outputImage = canvas.toDataURL(`image/${format}`, outputQuality);

            // Display processed image
            processedImg.src = outputImage;

            // Update progress
            updateProgress(100);

            // Enable download button
            downloadBtn.addEventListener('click', () => {
                const link = document.createElement('a');
                link.download = `removed-background.${format}`;
                link.href = outputImage;
                link.click();
            });

        } catch (error) {
            console.error('Error processing image:', error);
            showError('Error processing image. Please try again.');
        } finally {
            // Hide loading state
            setTimeout(() => {
                loadingOverlay.classList.remove('active');
                uploadProgress.classList.remove('active');
                progressBar.style.width = '0%';
            }, 500);
        }
    }

    function getSegmentationConfig(precision) {
        const configs = {
            high: {
                architecture: 'ResNet50',
                outputStride: 16,
                quantBytes: 4,
                internalResolution: 'high',
                segmentationThreshold: 0.9,
                maxDetections: 20,
                scoreThreshold: 0.3,
                nmsRadius: 20
            },
            medium: {
                architecture: 'MobileNetV1',
                outputStride: 16,
                multiplier: 1.0,
                quantBytes: 4,
                internalResolution: 'medium',
                segmentationThreshold: 0.7,
                maxDetections: 10,
                scoreThreshold: 0.3,
                nmsRadius: 20
            },
            low: {
                architecture: 'MobileNetV1',
                outputStride: 16,
                multiplier: 0.75,
                quantBytes: 2,
                internalResolution: 'low',
                segmentationThreshold: 0.5,
                maxDetections: 5,
                scoreThreshold: 0.3,
                nmsRadius: 20
            }
        };
        return configs[precision];
    }

    async function enhanceSegmentation(pixels, segmentation, precision) {
        const threshold = precision === 'high' ? 0.9 : precision === 'medium' ? 0.7 : 0.5;

        for (let i = 0; i < segmentation.length; i++) {
            const isBackground = segmentation[i] < threshold;
            if (isBackground) {
                pixels[i * 4 + 3] = 0; // Set alpha to 0 for background
            } else {
                // Enhance edges for better quality
                const x = (i % (pixels.length / 4)) * 4;
                const y = Math.floor(i / (pixels.length / 4)) * 4;
                
                if (x > 0 && x < pixels.length - 4 && y > 0 && y < pixels.length - 4) {
                    const surrounding = [
                        segmentation[i - 1],
                        segmentation[i + 1],
                        segmentation[i - pixels.length / 4],
                        segmentation[i + pixels.length / 4]
                    ];
                    
                    const hasBackgroundNeighbor = surrounding.some(val => val < threshold);
                    if (hasBackgroundNeighbor) {
                        pixels[i * 4 + 3] = Math.floor(pixels[i * 4 + 3] * 0.9); // Smooth edges
                    }
                }
            }
        }
    }

    function updateProgress(value) {
        progressBar.style.width = `${value}%`;
    }

    // Try Again Button
    tryAgainBtn.addEventListener('click', () => {
        uploadContainer.style.display = 'block';
        previewContainer.style.display = 'none';
        fileInput.value = '';
        originalImg.src = '';
        processedImg.src = '';
        progressBar.style.width = '0%';
    });
});
