/* global tf */
(async () => {
    const MODEL_URL = 'model.json';
    const IMAGE_SIZE = 224;
    const labels = ["Micro Plastic", "Clean Water"];

    const els = {
        btnLoad: document.getElementById('btn-load'),
        btnPredict: document.getElementById('btn-predict'),
        fileInput: document.getElementById('file'),
        previewImage: document.getElementById('preview'),
        statusText: document.getElementById('status'),
        resultText: document.getElementById('result')
    };

    let model = null;
    let modelLoaded = false;
    let imageSelected = false;

    function updateStatus(msg) {
        els.statusText.textContent = msg;
    }

    async function loadBackend() {
        await tf.ready();
    }

    async function loadModel() {
        if (model) return model;
        try {
            updateStatus('Loading backend...');
            await loadBackend();

            updateStatus('Loading model...');
            model = await tf.loadLayersModel(MODEL_URL, { fromTFHub: false });

            // Warm-up the model
            tf.tidy(() => {
                const warmTensor = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3], 'float32');
                model.predict(warmTensor);
            });

            modelLoaded = true;
            updateStatus('Model ready.');
            enablePrediction();
            return model;
        } catch (err) {
            console.error('Error loading model:', err);
            updateStatus('Failed to load model.');
            return null;
        }
    }

    function preprocessImage(img) {
        return tf.tidy(() => {
            let tensor = tf.browser.fromPixels(img);
            tensor = tf.image.resizeBilinear(tensor, [IMAGE_SIZE, IMAGE_SIZE]);
            tensor = tf.cast(tensor, 'float32');
            tensor = tf.div(tensor, tf.scalar(255));
            tensor = tensor.expandDims(0);
            return tensor;
        });
    }

    async function predict() {
        if (!modelLoaded) await loadModel();
        if (!els.previewImage.src) return;

        const inputTensor = preprocessImage(els.previewImage);
        let prediction = model.predict(inputTensor);

        // Print shape and raw prediction for debugging
        console.log("Prediction shape:", prediction.shape);
        prediction.print();

        // Get probabilities (raw prediction, softmax if multi-class)
        let probs;
        if (prediction.shape.length === 2 && prediction.shape[1] > 1) {
            const softmaxed = tf.softmax(prediction);
            probs = await softmaxed.data();
            softmaxed.dispose();
        } else {
            probs = await prediction.data();
        }

        // Display result as confidence percentages per label
        const displayText = Array.from(probs).map(
            (prob, index) => `${labels[index] || `Class ${index}`}: ${(prob * 100).toFixed(4)}%`
        ).join('\n');
        els.resultText.textContent = displayText;

        inputTensor.dispose();
        prediction.dispose();
    }

    function enablePrediction() {
        els.btnPredict.disabled = !(modelLoaded && imageSelected);
    }

    // Event listeners
    els.btnLoad.addEventListener('click', async () => {
        els.btnLoad.disabled = true;
        await loadModel();
    });

    els.fileInput.addEventListener('change', () => {
        const file = els.fileInput.files[0];
        if (!file) return;
        const url = URL.createObjectURL(file);
        els.previewImage.src = url;
        els.previewImage.onload = () => {
            imageSelected = true;
            els.previewImage.style.display = 'block';
            enablePrediction();
        };
    });

    els.btnPredict.addEventListener('click', predict);
})();
