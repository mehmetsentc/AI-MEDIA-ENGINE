# DAIDOI Studio frontend plan

This is an audit of the current static studio. The production interface was not redesigned. Publish Studio was not started. Write, Image Edit, Voice, Music, Video, Film, News, Publish, and Feed were not implemented.

## Current surface

The studio is one HTML document, `src/media_engine/studio/static/index.html`, with `studio.css` and `studio.js`. There is no React application. The title is still "AI-MEDIA-ENGINE Studio".

Navigation groups:

- Studio: Create, Projects
- Tools: Text, Voice, Images, Video, Music
- Library: Assets, History
- Settings and Account

The image inspector contains a prompt, the model label "Qwen Image", aspect ratios `16:9`, `9:16`, `1:1`, and `4:5`, a displayed resolution of 1024, a seed, Generate Image, Regenerate, and a download link. Variations stay disabled. Export stays disabled. A "Development data" label is shown when the controller is the development preview.

## Layout check

The development studio was opened at `http://127.0.0.1:8766` with `STUDIO_MODE=development` and without a Vast credential.

At a 1280-pixel width the app grid was `212px 748px 320px`. The sidebar navigation was a column. Document scroll width matched the viewport, so the page did not overflow sideways. Export was disabled. The download link was hidden. The visible page text had no credit balance.

At a narrow viewport the grid collapsed to one column, and the sidebar and navigation switched to a row. That matches the CSS breakpoints at 1180 pixels (icon sidebar and inspector drawer) and 720 pixels (single column and horizontal navigation). No sideways overflow was measured.

Clicking Images or Voice before a project exists leaves the inspector on the create brief. The image form and the "Coming next" copy are in the page, and they become the active inspector only after a scene asset is selected. That behavior was not changed.

## Reusable patterns

- A left navigation button with `data-view` switches the inspector.
- Scene assets are the unit of editing. An image asset stores prompt, seed, job id, and artifact id.
- `present_job` turns provider codes into short status titles. Capacity wait uses a waiting tone. Failure uses "Could not create the image".
- Empty, progress, and completed states hide or show Generate, Regenerate, and Download.
- Cost lines distinguish an estimate from an actual amount, and they stay hidden in development preview.
- Future tools reuse the same inspector and show "Coming next" rather than a working control.

## Minimum UI for a real image

A real generation still needs the existing image form, plus these production states that are missing or only partial:

- The signed-in user's credit balance, and the price of one image, before Generate is enabled.
- A job phase taken from `present_job`: preparing, starting the worker, loading the model, generating, saving, complete, or failed.
- A disabled Generate button while a job is running, so a double click cannot enqueue a second paid job.
- The finished image in the stage, loaded from the private artifact route, and a Download link that uses the expiring media URL.
- A failure title plus a plain refund line when settlement returns the reserved credits. The raw provider code stays out of the title.
- The development-data label must be absent in real mode. Real mode already refuses the preview engine.

## Credits, jobs, errors, and refunds

The API can return job status and error codes. The studio maps those through `present_job`. There is no balance widget and no refund sentence. Estimated and actual cost lines exist for a linked asset, and they are hidden while the studio is in development preview.

## Future product boundaries

These names may appear as navigation or copy. They are not operational products.

| Future product | Boundary |
| --- | --- |
| DAIDOI Write | Text view can remain a prompt editor. It must not claim a published writing engine. |
| DAIDOI Image | The only generation form that may call the image job API. |
| DAIDOI Image Edit | Not present. Do not add an edit action that submits a job. |
| DAIDOI Voice | Inspector copy stays "Coming next" until a voice engine and entitlement exist. |
| DAIDOI Music | Same coming-next boundary. Music scope controls are preferences only. |
| DAIDOI Video | Example briefs may mention video. The video tool must not submit a render job. |
| DAIDOI Film | No timeline renderer. The timeline view is organizational. |
| DAIDOI News | No newsroom ingestion or publish pipeline. |
| DAIDOI Publish | Export stays disabled. Do not start Publish Studio. |
| DAIDOI Feed | No feed reader or public posting. |

A new product should enter through its own entitlement, price, job type, and artifact kind. It should not reuse the image credit price or the Qwen worker.
