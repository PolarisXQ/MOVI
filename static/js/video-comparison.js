document.addEventListener('DOMContentLoaded', function () {
    initVideoComparisons();
});

function extractTaskAndVideo(container) {
    const baseVideo = container.querySelector('.video-wrapper video');
    const baseSource = baseVideo ? baseVideo.querySelector('source') : null;
    const overlaySource = container.querySelector('.video-overlay video source');

    const path = (baseSource && baseSource.getAttribute('src')) || (overlaySource && overlaySource.getAttribute('src')) || '';
    const taskMatch = path.match(/static\/demos\/([^/]+)\//);
    const taskName = taskMatch ? taskMatch[1] : '';

    let videoName = baseVideo && baseVideo.dataset ? baseVideo.dataset.baseName : '';
    if (!videoName) {
        const fileMatch = path.match(/\/([^/]+)\.mp4$/);
        videoName = fileMatch ? fileMatch[1] : '';
    }

    return { taskName: taskName, videoName: videoName };
}

function createMetaBox(title, body) {
    const box = document.createElement('div');
    box.className = 'meta-box';

    const heading = document.createElement('h4');
    heading.className = 'meta-box-title';
    heading.textContent = title;

    box.appendChild(heading);
    box.appendChild(body);
    return box;
}

function createPromptContent(taskName, videoName) {
    const promptText = document.createElement('pre');
    promptText.className = 'prompt-content';
    promptText.textContent = 'Loading prompt...';

    if (!taskName || !videoName) {
        promptText.textContent = 'Prompt not available.';
        return promptText;
    }

    const promptPath = 'static/demos/' + taskName + '/prompt/' + videoName + '.txt';
    fetch(promptPath)
        .then(function (response) {
            if (!response.ok) throw new Error('Prompt not found');
            return response.text();
        })
        .then(function (text) {
            promptText.textContent = text.trim() || '(empty prompt)';
        })
        .catch(function () {
            promptText.textContent = 'Prompt not found.';
        });

    return promptText;
}

function createGalleryContent(taskName, videoName) {
    const gallery = document.createElement('div');
    gallery.className = 'ref-gallery';

    const candidates = [videoName];
    if (videoName === 'car-roundabout') {
        candidates.push('car-roudabout');
    }

    for (let i = 0; i < 4; i += 1) {
        const image = document.createElement('img');
        image.loading = 'lazy';
        image.decoding = 'async';
        image.alt = videoName + ' reference view ' + i;
        image.dataset.idx = String(i);
        image.dataset.videoCandidates = candidates.join(',');
        image.dataset.taskName = taskName;

        const idx = String(i).padStart(2, '0');
        image.src = 'static/demos/' + taskName + '/ref/' + candidates[0] + '/' + candidates[0] + '_view' + idx + '.png';
        image.onerror = function () {
            const altCandidates = (this.dataset.videoCandidates || '').split(',');
            const current = this.dataset.currentCandidate || altCandidates[0];
            const currentIndex = altCandidates.indexOf(current);
            const nextIndex = currentIndex + 1;

            if (nextIndex >= altCandidates.length) {
                this.classList.add('ref-image-missing');
                this.onerror = null;
                return;
            }

            const nextCandidate = altCandidates[nextIndex];
            this.dataset.currentCandidate = nextCandidate;
            const viewId = String(this.dataset.idx).padStart(2, '0');
            this.src = 'static/demos/' + this.dataset.taskName + '/ref/' + nextCandidate + '/' + nextCandidate + '_view' + viewId + '.png';
        };

        gallery.appendChild(image);
    }

    return gallery;
}

function createModelContent(taskName, videoName) {
    const wrapper = document.createElement('div');
    wrapper.className = 'model-viewer-wrapper';

    const modelViewer = document.createElement('model-viewer');
    modelViewer.className = 'ref-model-viewer';
    modelViewer.setAttribute('camera-controls', '');
    modelViewer.setAttribute('touch-action', 'pan-y');
    modelViewer.setAttribute('shadow-intensity', '0.8');
    modelViewer.setAttribute('exposure', '1');
    modelViewer.alt = videoName + ' 3D model preview';

    const candidates = [videoName];
    if (videoName === 'car-roundabout') {
        candidates.push('car-roudabout');
    }

    modelViewer.dataset.videoCandidates = candidates.join(',');
    modelViewer.dataset.taskName = taskName;
    modelViewer.dataset.currentCandidate = candidates[0];
    modelViewer.src = 'static/demos/' + taskName + '/ref/' + candidates[0] + '/' + candidates[0] + '.glb';
    modelViewer.addEventListener('error', function () {
        const altCandidates = (this.dataset.videoCandidates || '').split(',');
        const current = this.dataset.currentCandidate || altCandidates[0];
        const currentIndex = altCandidates.indexOf(current);
        const nextIndex = currentIndex + 1;

        if (nextIndex >= altCandidates.length) return;
        const nextCandidate = altCandidates[nextIndex];
        this.dataset.currentCandidate = nextCandidate;
        this.src = 'static/demos/' + this.dataset.taskName + '/ref/' + nextCandidate + '/' + nextCandidate + '.glb';
    });

    wrapper.appendChild(modelViewer);
    return wrapper;
}

function createReferencePanel(taskName, videoName) {
    const panel = document.createElement('aside');
    panel.className = 'comparison-meta-panel';

    panel.appendChild(createMetaBox('3D Model', createModelContent(taskName, videoName)));
    panel.appendChild(createMetaBox('Reference Views', createGalleryContent(taskName, videoName)));
    panel.appendChild(createMetaBox('Prompt', createPromptContent(taskName, videoName)));
    return panel;
}

function initReferencePanels() {
    document.querySelectorAll('.comparison-card .video-compare-container').forEach(function (container) {
        const card = container.closest('.comparison-card');
        if (!card || card.querySelector('.comparison-meta-panel')) return;

        const info = extractTaskAndVideo(container);
        if (!info.taskName || !info.videoName) return;

        const panel = createReferencePanel(info.taskName, info.videoName);
        card.classList.add('comparison-card-with-meta');
        card.insertBefore(panel, container);
    });
}

function initControlSignalToggles() {
    document.querySelectorAll('.control-signal-toggle[data-demo-root][data-url-prefix]').forEach(function (btn) {
        const gridId = btn.getAttribute('data-demo-root');
        const urlPrefix = btn.getAttribute('data-url-prefix');
        const grid = document.getElementById(gridId);
        if (!grid || !urlPrefix) return;

        const baseVideos = grid.querySelectorAll('.video-wrapper video[data-base-name]');

        btn.addEventListener('click', function () {
            const useMask = btn.getAttribute('aria-pressed') !== 'true';
            btn.setAttribute('aria-pressed', useMask ? 'true' : 'false');

            baseVideos.forEach(function (video) {
                const container = video.closest('.video-compare-container');
                const overlayVideo = container ? container.querySelector('.video-overlay video') : null;
                const t = overlayVideo ? overlayVideo.currentTime : 0;
                const wasPlaying = !video.paused;

                const baseName = video.dataset.baseName;
                const maskName = video.dataset.maskBaseName || baseName;
                const source = video.querySelector('source');
                if (!baseName || !source) return;

                const dir = useMask ? 'edited_with_mask' : 'edited';
                const fileStem = useMask ? maskName : baseName;
                source.src = urlPrefix + '/' + dir + '/' + fileStem + '.mp4';
                video.load();

                function syncAfterLoad() {
                    video.currentTime = t;
                    if (overlayVideo) overlayVideo.currentTime = t;
                    if (wasPlaying) {
                        video.play().catch(function () {});
                    }
                }

                video.addEventListener('loadedmetadata', syncAfterLoad, { once: true });
            });
        });
    });
}

function initVideoComparisons() {
    initReferencePanels();

    const containers = document.querySelectorAll('.video-compare-container');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const video = entry.target.querySelector('.video-wrapper video');
                if (video) video.play().catch(e => console.log('Autoplay prevented:', e));
            } else {
                const video = entry.target.querySelector('.video-wrapper video');
                if (video) video.pause();
            }
        });
    }, { threshold: 0.1 });

    containers.forEach(container => {
        observer.observe(container);
        const slider = container.querySelector('.video-compare-slider');
        const overlay = container.querySelector('.video-overlay');
        const overlayVideo = overlay.querySelector('video');
        const baseVideo = container.querySelector('.video-wrapper video');

        let isDragging = false;

        // Sync videos
        baseVideo.onplay = () => overlayVideo.play();
        baseVideo.onpause = () => overlayVideo.pause();
        baseVideo.onseeking = () => overlayVideo.currentTime = baseVideo.currentTime;
        baseVideo.onseeked = () => overlayVideo.currentTime = baseVideo.currentTime;

        // Ensure overlay video matches size
        // We set the overlay video width to the container width so it aligns perfectly
        function resizeOverlayVideo() {
            overlayVideo.style.width = baseVideo.getBoundingClientRect().width + 'px';
            overlayVideo.style.height = baseVideo.getBoundingClientRect().height + 'px';
        }

        window.addEventListener('resize', resizeOverlayVideo);
        // Initial resize
        baseVideo.addEventListener('loadedmetadata', resizeOverlayVideo);
        // Fallback
        setTimeout(resizeOverlayVideo, 500);

        // Interaction
        function moveSlider(x) {
            const rect = container.getBoundingClientRect();
            let pos = (x - rect.left) / rect.width;

            // Clamp 0-1
            pos = Math.max(0, Math.min(1, pos));

            const percent = pos * 100;

            overlay.style.width = percent + '%';
            slider.style.left = percent + '%';
        }

        // Mouse events
        slider.addEventListener('mousedown', () => isDragging = true);
        window.addEventListener('mouseup', () => isDragging = false);
        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            moveSlider(e.clientX);
        });

        // Touch events
        slider.addEventListener('touchstart', () => isDragging = true);
        window.addEventListener('touchend', () => isDragging = false);
        window.addEventListener('touchmove', (e) => {
            if (!isDragging) return;
            moveSlider(e.touches[0].clientX);
        });

        // Click to jump
        container.addEventListener('click', (e) => {
            // Don't jump if clicking on controls (if generic controls are enabled)
            // But here we might not have standard controls visible or we might want custom ones.
            // For now, let's allow jumping if not dragging
            if (e.target !== slider && !slider.contains(e.target)) {
                moveSlider(e.clientX);
            }
        });
    });

    initControlSignalToggles();
}
