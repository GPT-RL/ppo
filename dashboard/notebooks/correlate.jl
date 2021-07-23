### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 6e3ef066-d115-11eb-2338-013a707dfe8a
begin
	using PlutoUI
	using DataFrames
	using Statistics
	using HTTP
	using JSON
	using Chain
	using Tables
	using CairoMakie
	using AlgebraOfGraphics
	CairoMakie.activate!(type = "svg")
end

# ╔═╡ f6396d66-cdd3-4fa5-afaa-7a43895250e0
begin
	using YAML
end

# ╔═╡ 41bd06a2-e7bd-46c6-9249-6e69223b0e11
begin
	HASRUA_ENDPOINT_URL = "http://hasura_graphql-engine_1:8080/v1/graphql"
	HASURA_ADMIN_SECRET = "hNuxFSTJMk28GMiFDpZDbviKtelFQamcb20UJiYZSIZ4LYsyLCCwQFYA6Y8HRSUo"
end;

# ╔═╡ 94185c13-b6d2-4337-b5ce-336f5e128032
function gql_query(query:: String; variables:: Dict = nothing)
	r = HTTP.request(
		"POST",
		HASRUA_ENDPOINT_URL;
		verbose=3,
		body= JSON.json(Dict("query" => query, "variables" => variables)) ,
		headers=Dict("x-hasura-admin-secret" => HASURA_ADMIN_SECRET),
	)
	@chain r begin
		_.body
		String
		JSON.parse(_, null=missing)
	    _["data"]
	end
end

# ╔═╡ ed56319c-fd7e-478e-a673-f799debbf7b3
function get_sweep(ids::Vector{Int64})::Vector{AbstractDict}
	query = """
query getSweep(\$ids: [Int!]!) {
  run(where: {sweep_id: {_in: \$ids}, run_logs: {id: {_is_null: false}}}) {
    run_logs(order_by: {id: desc}) {
      log
    }
    metadata
    id
  }
}
	"""
	gql_query(query; variables=Dict("ids" => ids))["run"]
end

# ╔═╡ da091e45-0d08-490a-ba85-ee9acb6700bc
sweep = get_sweep([17, 19, 22, 23])

# ╔═╡ c838c44c-4bd8-4eb0-a18b-ceaaa5bdce93
dicts = [
	Dict(
		"episode_return" => log["log"]["episode return"],
		"step" => log["log"]["step"],
		"run_id" => run_dict["id"],
		run_dict["metadata"]["parameters"]...
	) 
	for run_dict in sweep
		for log in run_dict["run_logs"]
			if 25000000 < log["log"]["step"] < 30000000
		]

# ╔═╡ 63ed4848-ca9a-48bb-bd02-35b4bf69d039
dframe = vcat(DataFrame.(dicts)...)

# ╔═╡ 2fdc2937-098a-4dfe-bba5-58a9164337e7
eltype(dframe[:, "subcommand"]) == String

# ╔═╡ 2735fbc7-a318-4b4e-8ceb-0e65084cf972
function filter_by_type(ty) 
	filter(name -> eltype(dframe[:, name]) == ty, names(dframe))
end

# ╔═╡ 1fe4e26a-2786-4094-a7e5-0154b461e874
begin
	grouped = groupby(dframe, :run_id)
	df = combine(grouped, 
		filter_by_type(Float64) .=> first,
		filter_by_type(Int64) .=> first,
		:episode_return .=> mean, 
		)
	sort!(df, [:episode_return_mean], rev=true)
	df
end

# ╔═╡ 490d6756-f769-4a13-9438-ad724aa013ee
begin
	bool_df = DataFrame()
	for name in names(df)
		if !occursin("episode_return", name)
			for value in df[:, name]
				new_name = string(replace(name, "_first"=>""), " = ", value)
				bool_df[:, :($new_name)] = df[:, name] .== value
			end
		end
	end
    bool_df.episode_return_mean = df.episode_return_mean
	bool_df
end

# ╔═╡ c3e89733-4a83-48d4-bf34-9f4f7895b798
function get_cor_df(df)
	return_index = findfirst(name -> name == "episode_return_mean", names(df))
	cor_mat = cor(Matrix(df))
	cor_df = DataFrame()
	cor_df.name = [replace(name, "_first" => "") for name in names(df)]
	cor_df.correlation = cor_mat[:, return_index]
	sort!(cor_df, [:correlation], )
	cor_df
end

# ╔═╡ f43f3cf4-b9e4-4c21-bf95-96a0c494c99b
md"# Continuous Correlation"

# ╔═╡ a4cb4dbe-b880-4a4a-9b91-ccf6eaff1b07
get_cor_df(df)

# ╔═╡ 5fb3a66b-49a2-41f6-aea5-65a2bd4d7240
md"# Discrete Correlation"

# ╔═╡ 6c27eed0-e73d-4a44-b286-dca6ac404dbd
get_cor_df(bool_df)

# ╔═╡ c1c3f1d7-6f54-460f-8f20-f73ed183f8fe
df[:, [:episode_return_mean, :run_id]]

# ╔═╡ fc6f061c-777f-4948-ae88-58467410cf68
run_ids = df[1:4, :run_id]

# ╔═╡ 3059d226-95e7-439d-9f3a-81836d18eb8a
parameters = [
	run_dict["metadata"]["parameters"]
	for run_dict in sweep
		if run_dict["id"] in run_ids
		]


# ╔═╡ Cell order:
# ╟─6e3ef066-d115-11eb-2338-013a707dfe8a
# ╟─41bd06a2-e7bd-46c6-9249-6e69223b0e11
# ╠═94185c13-b6d2-4337-b5ce-336f5e128032
# ╠═ed56319c-fd7e-478e-a673-f799debbf7b3
# ╠═da091e45-0d08-490a-ba85-ee9acb6700bc
# ╠═c838c44c-4bd8-4eb0-a18b-ceaaa5bdce93
# ╟─63ed4848-ca9a-48bb-bd02-35b4bf69d039
# ╟─2fdc2937-098a-4dfe-bba5-58a9164337e7
# ╟─2735fbc7-a318-4b4e-8ceb-0e65084cf972
# ╠═1fe4e26a-2786-4094-a7e5-0154b461e874
# ╟─490d6756-f769-4a13-9438-ad724aa013ee
# ╠═c3e89733-4a83-48d4-bf34-9f4f7895b798
# ╠═f43f3cf4-b9e4-4c21-bf95-96a0c494c99b
# ╟─a4cb4dbe-b880-4a4a-9b91-ccf6eaff1b07
# ╠═5fb3a66b-49a2-41f6-aea5-65a2bd4d7240
# ╟─6c27eed0-e73d-4a44-b286-dca6ac404dbd
# ╠═c1c3f1d7-6f54-460f-8f20-f73ed183f8fe
# ╠═fc6f061c-777f-4948-ae88-58467410cf68
# ╠═3059d226-95e7-439d-9f3a-81836d18eb8a
# ╠═f6396d66-cdd3-4fa5-afaa-7a43895250e0
