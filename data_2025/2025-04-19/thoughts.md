# Thoughts

There's a continuous range for outliers.
Set a swelling limit to identify outliers.
The limit can come from Robinson correlation.
Look at the parameter distributions of outliers.

> Robinson correlation is too restricting.
> The parameter distribution of outliers don't provide insight.
> Actually, there are correlations.
> I need to change the cutoff for the outliers.
> Can we plot the normal/outlier points in different colors?
> Maybe use a slider to see the effect of the cutoff? Good idea!

If there is a correlation among the outliers, good.
Otherwise, we'd have to perform down selection.
I need to do a literature search anyway.

> If there are correlations, how can we put a limit on params?
> These limits need to be convenient for surrogate building.
> Any cutoff above Robinson upper limit is fair game.
> We need insights for both normal and outlier datapoints.


How to combine the data from different operational conditions?
Two different fission rates, grain sizes, and fuel temperatures.

It's a good idea to analyze them individually first.

What about the problem statement for IUQ?
Should I average all the swelling results from different conditions?
The swelling correlation comes from the average of swelling results
from different operational conditions as well.
We need to make the problem statement rigorous.
