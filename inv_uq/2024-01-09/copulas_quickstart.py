from copulas.datasets import sample_trivariate_xyz

data = sample_trivariate_xyz()
print(data.head())

from copulas.visualization import scatter_3d

fig1 = scatter_3d(data)
fig1.show()

from copulas.multivariate import GaussianMultivariate

copula = GaussianMultivariate()
copula.fit(data)

num_samples = 1000

synthetic_data = copula.sample(num_samples)
print(synthetic_data.head())

from copulas.visualization import compare_3d

fig2 = compare_3d(data, synthetic_data)
fig2.show()

## saving the model
# copula.save(model_path)

## loading the model
# new_copula = GaussianMultivariate.load(model_path)

copula_params = copula.to_dict()
#print(copula_params)

## another way of loading
# new_copula = GaussianMultivariate.from_dict(copula_params)
