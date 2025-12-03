from flask import render_template


def render_main_page(video_descriptions, used_videos):
    if not video_descriptions:
        return render_template('index.html', frames=[], used_videos={})

    return render_template(
        'index.html',
        frames=video_descriptions,
        used_videos=used_videos
    )
