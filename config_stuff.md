## Note

If ray tune is used, any configuration provided by ray tune that conflicts with existing configurations will be overridden by ray tune. For instance, your default environment frame stacking could be 4, but if you are hyparameter tuning using ray to try and find a better frame stacking number, the number returned by ray will override it.


# Configuration Specifics

## Env

This category handles anything to do with our environment. Custom rewards, number of vectorized environments, frame stacking, novice intialization, etcetc.











