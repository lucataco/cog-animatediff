# guoyww/AnimateDiff Cog model

This is an implementation of [guoyww/AnimateDiff](https://github.com/guoyww/animatediff/) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog build -t animatediff

Then, you can run predictions:

    cog predict -i prompt="masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, outdoors, falling petals, white hair, brown eyes"

## Example Output

Example output for prompt: "masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, outdoors, falling petals, white hair, brown eyes"

![alt text](output.gif)
