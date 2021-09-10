### A Pluto.jl notebook ###
# v0.15.1

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
	using MLStyle
end

# ╔═╡ 41bd06a2-e7bd-46c6-9249-6e69223b0e11
HASRUA_ENDPOINT_URL = "http://rldl12.eecs.umich.edu:1200/v1/graphql"

# ╔═╡ 94185c13-b6d2-4337-b5ce-336f5e128032
function gql_query(query:: String; variables:: Dict = nothing)
	r = HTTP.request(
		"POST",
		HASRUA_ENDPOINT_URL;
		verbose=3,
		body= JSON.json(Dict("query" => query, "variables" => variables)) ,
		# headers=Dict("x-hasura-admin-secret" => HASURA_ADMIN_SECRET),
	)
	@chain r begin
		_.body
		String
		JSON.parse(_, null=missing)
	    _["data"]
	end
end

# ╔═╡ ed56319c-fd7e-478e-a673-f799debbf7b3
function get_sweep(sweep_ids::AbstractVector{Int}, max_step::Int)
	query = """
		query getSweepRuns(\$ids: [Int!], \$max_step: Int!) {
		  logs_less_than_step(args: {max_step: \$max_step}, where: {run: {sweep_id: {_in: \$ids}}}) {
			log
			run_id
			run {
			  metadata
			  sweep_id
			}
		  }
		}
  	""" 
	@chain query begin
		gql_query(_; variables=Dict("ids" => sweep_ids, "max_step" => max_step))
		_["logs_less_than_step"]		
	map(d -> Dict(
				"run_id" => d["run_id"],
				"sweep_id" => d["run"]["sweep_id"],
				d["log"]...,
				d["run"]["metadata"]["parameters"]...,
				"host_machine" => d["run"]["metadata"]["host_machine"],
				), _)

		map(d -> Dict(
				d...,
				[k => v for (k1, v1, k2, v2) in [
							(
								"hours", get(d, "time-delta", 0) / 3600, 
								"time-delta", get(d, "hours", 0) * 3600,
							),
							(
								"env", get(d, "env_name", nothing),
								"env_name", get(d, "env", nothing),
							)
						] 
						for (k, v) in [
								(k1, get(d, k1, v1)), 
								(k2, get(d, k2, v2)),
								]]...,				
				[name => get(d, name, false) for name in [
						]]...,
				[name => get(d, name, nothing) for name in [
							"action loss", 
							"entropy", 
							"episode return", 
							"fps", 
							"gradient norm", 
							"randomize_parameters",
							"save count",
							"value loss",
							"test episode return",
							"train_wpe",
							"train_ln",
							"data_parallel",
							"linguistic_analysis_save_interval"
						]]...,
				), _)
			vcat(DataFrame.(_)...)
	
	end
end;

# ╔═╡ da091e45-0d08-490a-ba85-ee9acb6700bc


# ╔═╡ 16afb1df-73f6-483b-9d4b-eb07aaa877e0
SWEEP= 1079

# ╔═╡ c1fdd3b8-6aa3-4cd3-a381-07ab41753e96
MIN_STEP = 3000000

# ╔═╡ 1a6b7810-977d-4a82-ab14-b24e154ad491
EPISODE_RETURN = "episode return"

# ╔═╡ 1fe4e26a-2786-4094-a7e5-0154b461e874
df = @chain get_sweep([SWEEP], 1000000000) begin
	filter(row -> row["step"] > MIN_STEP, _)
	filter(row -> !isnothing(row[EPISODE_RETURN]), _)
	groupby(_, "run ID")
	combine(_, [
			[(name .=> first .=> name) for name in names(_) if name != EPISODE_RETURN]...
			EPISODE_RETURN .=> mean .=> EPISODE_RETURN
		]
		)
	_[!, [n for n in names(_) if length(unique(_[!, n])) > 1]]
	sort!(_, [EPISODE_RETURN], rev=true)
	# rename(name -> replace(name, "_first" => ""), _)
end

# ╔═╡ 8b15faf7-0d54-481f-a513-311e963dd8c1
Dict(k => unique(df, k)[!, k] for k in names(df))

# ╔═╡ 490d6756-f769-4a13-9438-ad724aa013ee
begin
	bool_df = DataFrame()
	for name in names(df)
		if !(name in [EPISODE_RETURN, "action loss", "entropy", "fps", "gradient norm", "hours", "time", "time-delta", "value loss", "save count"])
			for value in df[:, name]
				new_name = string(name, " = ", value)
				bool_df[:, :($new_name)] = df[:, name] .== value
			end
		end
	end
    bool_df[!, EPISODE_RETURN] = df[!, EPISODE_RETURN]
	bool_df
end

# ╔═╡ c3e89733-4a83-48d4-bf34-9f4f7895b798
function get_cor_df(df)

	cor_df = DataFrame()
	
	cor_df.name = names(df)
	
	cor_mat = cor(Matrix(df))
	return_index = findfirst(name -> name == EPISODE_RETURN, names(df))
	return_index
	cor_df.correlation = cor_mat[:, return_index]
	
	sort!(cor_df, [:correlation], )
	cor_df
end

# ╔═╡ 6c27eed0-e73d-4a44-b286-dca6ac404dbd
get_cor_df(bool_df)

# ╔═╡ c422048a-d149-4935-aa33-7684c94dd847
get_cor_df(bool_df)

# ╔═╡ 7e2ac3d5-9b72-4859-96a7-b6b8711e14b3
min_returns = @chain sweeps begin
	dropmissing(_, EPISODE_RETURN)
	# groupby(_, [:env])
	combine(_, EPISODE_RETURN => minimum)
	Dict(k=>v for (k,v) in eachrow(_))
end

# ╔═╡ 5a29110f-deca-4812-9240-ab445ed665c8
max_returns = Dict(
	"BeamRider-v0" => 1590, 
	"PongNoFrameskip-v0" => 20.7, 
	"Seaquest-v0" => 1204.5,
	"Qbert-v0" => 14293.3
) # from PPO paper

# ╔═╡ c838c44c-4bd8-4eb0-a18b-ceaaa5bdce93
dframe = @chain sweeps begin
	# filter(:step => >=(8000000), _)
	# groupby(_, [:env])
	# transform(_, ["env", EPISODE_RETURN] =>
	# 	function (envs, ret) 
	# 		@match [Set(envs)...] (
	# 			[env] => (ret .- min_returns[env]) ./ max_returns[env] 							)
	# 	end => [EPISODE_RETURN])
	_[!, filter(names(_)) do name
			!(name in [
				"action loss",
				"config",
				"cuda",
				"entropy",
				"eval_interval",
				"fps",
				"gradient norm",
				"hours",
				"log_interval",
				"log_level",
				"num_env_steps",
				"recurrent_policy",
				"run_id",
				"save_interval",
				"save_path",
				"step",
				"subcommand",
				"sweep_id",
				"time", 
				"time-delta",
				"value loss",
			])
		end]
end

# ╔═╡ 2735fbc7-a318-4b4e-8ceb-0e65084cf972
function filter_by_type(ty) 
	filter(name -> eltype(dframe[:, name]) == ty, names(dframe))
end

# ╔═╡ f43f3cf4-b9e4-4c21-bf95-96a0c494c99b
md"# Continuous Correlation"

# ╔═╡ a4cb4dbe-b880-4a4a-9b91-ccf6eaff1b07
get_cor_df(df)

# ╔═╡ 5fb3a66b-49a2-41f6-aea5-65a2bd4d7240
md"# Discrete Correlation"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[compat]
Chain = "~0.4.8"
DataFrames = "~1.2.2"
HTTP = "~0.9.13"
JSON = "~0.21.2"
MLStyle = "~0.4.10"
PlutoUI = "~0.7.9"
Tables = "~1.5.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Chain]]
git-tree-sha1 = "cac464e71767e8a04ceee82a889ca56502795705"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.4.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "727e463cfebd0c7b999bbf3e9e7e16f254b94193"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.34.0"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "44e3b40da000eab4ccb1aecdc4801c040026aeb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.13"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MLStyle]]
git-tree-sha1 = "594e189325f66e23a8818e5beb11c43bb0141bcd"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.10"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "2ca267b08821e86c5ef4376cffed98a46c2cb205"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═6e3ef066-d115-11eb-2338-013a707dfe8a
# ╠═41bd06a2-e7bd-46c6-9249-6e69223b0e11
# ╠═94185c13-b6d2-4337-b5ce-336f5e128032
# ╠═ed56319c-fd7e-478e-a673-f799debbf7b3
# ╠═da091e45-0d08-490a-ba85-ee9acb6700bc
# ╟─6c27eed0-e73d-4a44-b286-dca6ac404dbd
# ╠═1fe4e26a-2786-4094-a7e5-0154b461e874
# ╠═16afb1df-73f6-483b-9d4b-eb07aaa877e0
# ╠═c1fdd3b8-6aa3-4cd3-a381-07ab41753e96
# ╠═c422048a-d149-4935-aa33-7684c94dd847
# ╠═8b15faf7-0d54-481f-a513-311e963dd8c1
# ╠═490d6756-f769-4a13-9438-ad724aa013ee
# ╠═c3e89733-4a83-48d4-bf34-9f4f7895b798
# ╠═2735fbc7-a318-4b4e-8ceb-0e65084cf972
# ╠═1a6b7810-977d-4a82-ab14-b24e154ad491
# ╠═7e2ac3d5-9b72-4859-96a7-b6b8711e14b3
# ╠═5a29110f-deca-4812-9240-ab445ed665c8
# ╠═c838c44c-4bd8-4eb0-a18b-ceaaa5bdce93
# ╟─f43f3cf4-b9e4-4c21-bf95-96a0c494c99b
# ╠═a4cb4dbe-b880-4a4a-9b91-ccf6eaff1b07
# ╟─5fb3a66b-49a2-41f6-aea5-65a2bd4d7240
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
