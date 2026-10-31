# guoyww/AnimateDiff Cog model

[![Try a demo on Replicate](https://replicate.com/lucataco/animate-diff/badge)](https://replicate.com/lucataco/animate-diff)
https://github.com/KIMGEONUNG/BigColor 

This is an implementation of [guoyww/AnimateDiff](https://github.com/guoyww/animatediff/) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog build -t animatediff

Then, you can run predictions:

    cog predict -i prompt="masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, outdoors, falling petals, white hair, brown eyes"

## Example Output

Example output for prompt: "masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, outdoors, falling petals, white hair, brown eyes"

![alt text](output.gif)

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

The model weights are not covered by this license. The [AnimateDiff motion modules](https://huggingface.co/guoyww/animatediff) are Apache 2.0, the Stable Diffusion 1.5 base model is under CreativeML Open RAIL-M, and each bundled Civitai checkpoint follows its own Civitai license.
