import os

# Configuration
DEMOS_DIR = "static/demos"
OUTPUT_FILE = "generated_demos.html"

def generate_html():
    html_output = ""
    
    # Get all subdirectories in demos dir
    if not os.path.exists(DEMOS_DIR):
        print(f"Error: {DEMOS_DIR} does not exist.")
        return ""

    categories = [d for d in os.listdir(DEMOS_DIR) if os.path.isdir(os.path.join(DEMOS_DIR, d)) and not d.startswith('.')]
    categories.sort()

    for category in categories:
        cat_path = os.path.join(DEMOS_DIR, category)
        edited_path = os.path.join(cat_path, "edited")
        origin_path = os.path.join(cat_path, "origin")

        if not os.path.exists(edited_path) or not os.path.exists(origin_path):
            continue

        html_output += f'<h3 class="title is-4 has-text-centered" style="color: #4a4a4a;">{category.capitalize()}</h3>\n'
        html_output += '<div class="columns is-multiline is-centered">\n'

        # Find matching videos
        edited_videos = [f for f in os.listdir(edited_path) if f.endswith(('.mp4', '.mov', '.webm')) and not f.startswith('.')]
        edited_videos.sort()

        for video_file in edited_videos:
            origin_video_file = os.path.join(origin_path, video_file)
            
            # Check if corresponding origin video exists
            if os.path.exists(origin_video_file):
                
                # Relative paths for HTML
                rel_edited = os.path.join(DEMOS_DIR, category, "edited", video_file)
                rel_origin = os.path.join(DEMOS_DIR, category, "origin", video_file)

                html_output += f'''
        <div class="column is-half">
            <div class="video-compare-container">
                <div class="video-wrapper">
                    <video autoplay loop muted playsinline poster="">
                        <source src="{rel_edited}" type="video/mp4">
                    </video>
                </div>
                <div class="video-overlay">
                    <video autoplay loop muted playsinline poster="">
                        <source src="{rel_origin}" type="video/mp4">
                    </video>
                </div>
                <div class="video-compare-slider">
                    <div class="slider-handle"></div>
                </div>
                <span class="compare-label label-left">Original</span>
                <span class="compare-label label-right">Edited</span>
            </div>
        </div>
'''
        
        html_output += '</div>\n<br>\n'

    return html_output

if __name__ == "__main__":
    generated_html = generate_html()
    print(generated_html)
    
