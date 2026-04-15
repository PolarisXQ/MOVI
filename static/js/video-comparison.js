document.addEventListener('DOMContentLoaded', function () {
    initVideoComparisons();
});

function initInstanceInsControlSignalToggle() {
    const btn = document.getElementById('control-signal-toggle');
    const grid = document.getElementById('instance-ins-demos');
    if (!btn || !grid) return;

    const baseVideos = grid.querySelectorAll('.video-wrapper video[data-base-name]');

    btn.addEventListener('click', function () {
        const useMask = btn.getAttribute('aria-pressed') !== 'true';
        btn.setAttribute('aria-pressed', useMask ? 'true' : 'false');

        baseVideos.forEach(function (video) {
            const container = video.closest('.video-compare-container');
            const overlayVideo = container ? container.querySelector('.video-overlay video') : null;
            const t = overlayVideo ? overlayVideo.currentTime : 0;
            const wasPlaying = !video.paused;

            const name = video.dataset.baseName;
            const source = video.querySelector('source');
            if (!name || !source) return;

            const dir = useMask ? 'edited_with_mask' : 'edited';
            source.src = 'static/demos/instance_ins/' + dir + '/' + name + '.mp4';
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
}

function initVideoComparisons() {
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

    initInstanceInsControlSignalToggle();
}
