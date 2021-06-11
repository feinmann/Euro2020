### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 5d561ec1-3a9a-44dd-b1d9-dc2622e10b54
begin
	using CSV
	using DataFrames
	using Distributions
	using Plots
	using LinearAlgebra
	using StatsBase
end

# ╔═╡ e6a2e856-ad88-4425-86ca-098359a0210c
md"# EURO 2020 Kicktipp optimization"

# ╔═╡ da2b04dc-8177-48d7-b3cc-89d16444a8f2
md"### Type definitions"

# ╔═╡ f17cab32-c91b-11eb-1d42-2de9769ace13
begin
	"Representation of a team taking part in the tournament"
	struct Team
		name::String
		"The team's Elo number (https://en.wikipedia.org/wiki/Elo_rating_system)"
		elo::Int64
	end
	
	
	"Representation of a tournament group (Group A, B, C, D, E, F)"
	struct Group
		label::String
		teams::Vector{Team}
	end
	
	
	"A single match between two teams"
	struct Match
		team1::Team
		team2::Team
		group_stage::Bool
	end
	
	
	"The result of a single match"
	struct MatchResult
		match::Match
		"Number of goals scored by match.team1"
		score_team1::Int64
		"Number of goals scored by match.team2"
		score_team2::Int64
		"Whether match went into extra time"
		extra_time::Bool
		"Whether match went into penalty shootout"
		penalty::Bool
	end
	
	
	"Probabilities for different results of a single match"
	struct MatchProbabilities
		match::Match
		"""
		Result probabilities, e.g 
		
		* p[1, 2] is the probability for team2 winning 0:1
		* p[3, 2] is the probability for team1 winning 2:1
		
		etc...
		"""
		p::Array{Float64, 2}
	end
	
	
	"""
	Result of an optimization for a single match, i.e. the score
	that yields the highest expected points in a 4-3-2-Kicktipp.
	"""
	struct BestMatchResult
		match::Match
		"Number of goals scored by team1 that yields the highest number of points"
		score_team1::Int64
		"Number of goals scored by team2 that yields the highest number of points"
		score_team2::Int64
		"""
		Reward matrix, e.g.
		expected_rewards[4, 2] is the expected number of points scored in
		a 4-3-2-Kicktipp if betting on team1 winning 3:1
		"""
		expected_rewards::Array{Float64, 2}
		"""
		Result probabilities, e.g 
		
		* score_probabilities[1, 2] is the probability for team2 winning 0:1
		* score_probabilities[3, 2] is the probability for team1 winning 2:1
		
		etc...
		"""
		score_probabilities::Array{Float64, 2}
	end
	
	
	"""
	Result of a full tournament
	"""
	struct TournamentResult
		winner::Team
		semifinalists::Set{Team}
		group_winners::Vector{Team}
	end
	
	
	struct BestTournamentResult
		winner::Pair{Team, Float64}
		semifinalists::Pair{Set{Team}, Float64}
		group_winners::Vector{Pair{Team, Float64}}
	end
end

# ╔═╡ 5c24a3cf-9fd0-4aa3-8fa4-d415bdfcd588
"List of matches in group"
function group_matches(group)
	[
		Match(group.teams[1], group.teams[2], true),
		Match(group.teams[3], group.teams[4], true),
		Match(group.teams[1], group.teams[3], true),
		Match(group.teams[2], group.teams[4], true),
		Match(group.teams[4], group.teams[1], true),
		Match(group.teams[2], group.teams[3], true),
	]
end	

# ╔═╡ 96f0a2a0-c5cb-4d04-9d38-47269fd58308
"Initialize list of groups based on group dataframe"
function make_groups(df)
	groups = []
	for (g_label, g_data) in pairs(groupby(df, :Group))
		teams = [
			Team(country, elo)
			for (country, elo)
			in eachrow(g_data[:, [:Country, :Elo]])
		]
		push!(groups, Group(g_label.Group, teams))
	end
	groups
end

# ╔═╡ d1682178-0dd6-413b-84e9-db90afd1b309
"Average goal rates (expected number of goals per match) for team1 and team2 in a match"
function score_rates(match, playtime)
	alpha = 0.1727
    beta = 0.107132
	elo_diff = match.team1.elo - match.team2.elo
	λ1 = exp(alpha + beta*(elo_diff / 100)) * playtime / 90
	λ2 = exp(alpha - beta*(elo_diff / 100)) * playtime / 90
	λ1, λ2
end

# ╔═╡ 1557b1cd-6596-426f-a1d2-e3d82a5a974d
md"#### Example: Score rates of match between best and worst team"

# ╔═╡ c1dee168-93bd-4640-b728-c54dfdb387a8
score_rates(Match(Team("Belgium", 2107), Team("North Macedonia", 1600), false), 90)

# ╔═╡ e6dd90d9-80eb-4238-bf42-ce52d57f8b08
"Simulate a single match"
function play_match(match)
	# Taken from https://github.com/msternke/UEFA-Euro-2021-sim
	# Note: pretty sure this underestimates the dispersion
	alpha = 0.1727
    beta = 0.107132
	penalty_success_rate = 0.75
	extra_time = false
	penalty = false
	elo_diff = match.team1.elo - match.team2.elo
	λ1, λ2 = score_rates(match, 90)
	score1 = rand(Poisson(λ1))
	score2 = rand(Poisson(λ2))
	if !match.group_stage & (score1 == score2)
		extra_time = true
		λ1, λ2 = score_rates(match, 30)
		score1 += rand(Poisson(λ1))
		score2 += rand(Poisson(λ2))

		if score1 == score2
			penalty = true
			n_penalties = 0
			while (n_penalties < 5) | (score1 == score2)
				# Penalty shootout
				score1 += rand(Bernoulli(penalty_success_rate))
				score2 += rand(Bernoulli(penalty_success_rate))
				n_penalties += 1
			end
		end
	end
	MatchResult(match, score1, score2, extra_time, penalty)
end

# ╔═╡ b151cc55-a507-4f69-b548-555a2be3aba5
md"#### Example: Play a single match"

# ╔═╡ 6481fec5-04cf-4bf0-8330-3c7bffe129c3
play_match(Match(Team("Turkey", 1809), Team("Italy", 2008), false))

# ╔═╡ adc93b5b-da90-4e02-aad1-38d21690e432
"Estimate probabilities of final scores for a given match by simulating `n_samples` repetitions of the match"
function match_probabilities(match, size, n_samples=10_000)
	p = zeros(Float64, size, size)
	for _ = 1:n_samples
		result = play_match(match)
		p[min(result.score_team1 + 1, size), min(result.score_team2 + 1, size)] += 1
	end
	p /= n_samples
	MatchProbabilities(match, p)
end

# ╔═╡ d4f9a8db-b796-47c5-b485-5720371d4460
md"#### Example match probabilities"

# ╔═╡ ce8e24a6-6fed-483b-95df-9b2cc42bda4f
match_probabilities(Match(Team("Turkey", 1809), Team("Italy", 2008), false), 10, 100_000)

# ╔═╡ dc601549-1fd6-4fab-bbc6-6bc5f0305f68
"""
Matrix containing the 4-3-2-Kicktipp reward, assuming the actual result of the match
was i-1:j-1
e.g. m[1, 1] (for any i, j) contains the Kicktipp reward for the tipp 0:0

size - 1 is the highest number of goals per team that is considered
"""
function reward_matrix(i, j, size)
	m = zeros(size, size)
	for k = 1:size
		for l = 1:size
			m[k, l] = 2 * Int(sign(k - l) == sign(i - j))
		end
	end
	m += diagm(j - i => ones(size - abs(j - i)))
	m[i, j] += 1
	m
end

# ╔═╡ d434f991-2ed5-4d3e-bd5c-bf346ac8163d
md"""
#### Example reward matrix

Example reward matrix if the actual match ended 2:5
"""

# ╔═╡ 5e48f29f-cb09-4ca7-86c6-e17cc0278e90
reward_matrix(3, 6, 8)

# ╔═╡ 299f4b3c-b83e-4b8a-80c9-e41a96dda227
"""
For each match in the group stage, return the tipp that yields the highest expected 
reward.
"""
function optimize_groupstage(groups, size=8, n_samples=10_000)
	results = Array{BestMatchResult, 2}(undef, length(groups), 6)
	for (g_i, group) in enumerate(groups)
		for (m_i, match) in enumerate(group_matches(group))
			p = match_probabilities(match, size, n_samples).p
			e = Array{Float64, 2}(undef, size, size)
			for i = 1:size
				for j = 1:size
					e[i, j] = sum(p .* reward_matrix(i, j, size))
				end
			end
			best = argmax(e)
			results[g_i, m_i] = BestMatchResult(match, best[1] - 1, best[2] - 1, e, p)
		end
	end
	results
end	

# ╔═╡ f6c227bb-4f7e-4db1-9ebb-0bcab5977f9c
function play_groupstage(groups)
	results = Array{MatchResult, 2}(undef, length(groups), 6)
	for (g_i, group) in enumerate(groups)
		for (m_i, match) in enumerate(group_matches(group))
			results[g_i, m_i] = play_match(match)
		end
	end
	results
end

# ╔═╡ 3cd0ccba-f12d-4d8d-a049-091916813816
function table(group_result)
	teams = union(
		Set(r.match.team1 for r in group_result), 
		Set(r.match.team1 for r in group_result)
	)
	points = Dict(zip(teams, 0 for _ in 1:length(teams)))
	goaldiff = Dict(zip(teams, 0 for _ in 1:length(teams)))
	for match_result in group_result
		if match_result.score_team1 > match_result.score_team2
			p1 = 3; p2 = 0
		elseif match_result.score_team1 < match_result.score_team2
			p1 = 0; p2 = 3
		else
			p1 = 1; p2 = 1
		end
		points[match_result.match.team1] += p1
		points[match_result.match.team2] += p2
		goaldiff[match_result.match.team1] += match_result.score_team1 - match_result.score_team2
		goaldiff[match_result.match.team2] += match_result.score_team2 - match_result.score_team1
	end
	df_points = DataFrame([[t for t in keys(points)], [p for p in values(points)]], [:Team, :Points])
	df_goaldiff = DataFrame([[t for t in keys(goaldiff)], [p for p in values(goaldiff)]], [:Team, :Goaldiff])
	sort(innerjoin(df_points, df_goaldiff, on=:Team), [:Points, :Goaldiff], rev=true)
end

# ╔═╡ 00c323a6-fa14-4ac3-b207-89ca48f7ff47
winner(mr::MatchResult) = mr.score_team1 > mr.score_team2 ? mr.match.team1 : mr.match.team2

# ╔═╡ 5cd4fa9d-9d0a-434b-91a8-6b5c4f9ebe2b
function quarter_matches(last_16_results)
	matches = [
		Match(winner(last_16_results[1]), winner(last_16_results[2]), false),
		Match(winner(last_16_results[3]), winner(last_16_results[4]), false),
		Match(winner(last_16_results[5]), winner(last_16_results[6]), false),
		Match(winner(last_16_results[7]), winner(last_16_results[8]), false),
	]
end

# ╔═╡ 006dd1f9-2dea-4481-8951-ffa1bd21dd7a
function semi_matches(quarter_results)
	matches = [
		Match(winner(quarter_results[1]), winner(quarter_results[2]), false),
		Match(winner(quarter_results[3]), winner(quarter_results[4]), false),
	]
end

# ╔═╡ c7f66257-ba75-4815-a75b-59a9d616c532
function final_match(semi_results)
	Match(winner(semi_results[1]), winner(semi_results[2]), false)
end

# ╔═╡ 695793a4-af27-4cce-9d70-883d9c8d650d
function best_semifinal(semifinal_counts)
	n = 0
	expectations = Dict(p[1] => 0 for p in semifinal_counts)
	for (semifinalists, count) in semifinal_counts
		n += count
		for k in keys(expectations)
			expectations[k] += length(intersect(k, semifinalists)) * count
		end
	end
	findmax(Dict(k => v / n  for (k, v) in expectations)) |> reverse
end

# ╔═╡ de6ff9b0-97d9-46f0-878b-8c938cbea904
md"### Download team/group data into dataframe"

# ╔═╡ d1f6466a-67ed-4b23-bd05-80db18792621
team_data = download("https://raw.githubusercontent.com/msternke/UEFA-Euro-2021-sim/master/data/euro_teams_data.csv") |> CSV.File |> DataFrame

# ╔═╡ 05f595ea-18a4-4a3a-8efc-cbd8f00be497
md"#### Make group metadata"

# ╔═╡ 6c7fdb33-f5f9-4093-b529-9c5a3d1911a8
groups = make_groups(team_data)

# ╔═╡ 294a097f-043e-4a2c-b7a3-d88177975ccd
play_groupstage(groups)

# ╔═╡ bf0cb9cf-ed9e-4cf5-8d14-cd8aa47d1ad9
typeof(table(play_groupstage(groups)[1, :]))

# ╔═╡ 1950efd1-1da8-4d8c-8403-5341a8c1d364
md"#### Run optimization for group stage"

# ╔═╡ 40461daa-c6a3-4a5e-bd71-539db7fea679
results = optimize_groupstage(groups, 10, 100_000)

# ╔═╡ 19a8ce2b-46d4-4a10-b422-46f8b6b07443
md"#### Short overview of results with highest reward"

# ╔═╡ b74b47bd-0ac9-4e04-8201-08375e39e55a
[(r.score_team1, r.score_team2) for r in results]

# ╔═╡ 664a812f-97f6-419a-bc1c-9e44a6c32456
md"#### Short overview of most probable results"

# ╔═╡ c2565dc1-7d05-44da-86e8-0eb020461403
[Tuple(argmax(r.score_probabilities)) .- 1 for r in results]

# ╔═╡ df720e43-ac9b-4dce-9614-e17ce81c2042
md"#### Results of first round only"

# ╔═╡ 09de1468-20f1-4cef-8fb5-07a989a6e163
reshape(results[:, 1:2], :)

# ╔═╡ 74298dfd-4ccb-45b2-977c-25452f3155f9
md"#### Run simulation for whole tournament"

# ╔═╡ e927a22e-257e-41ba-bba3-d721cfa05b94
function thirds_groups(group_tables)
	thirds = DataFrame(t[3, :] for t in group_tables)
	thirds[!, :Group] = collect(1:6)
	sort(sort(thirds, [:Points, :Goaldiff], rev=true)[1:4, :Group])
end

# ╔═╡ 61a11c9d-cddb-44c0-aaa1-619cc650e208
function opponent_1B(group_tables)
	mapping = Dict(
		[1, 2, 3, 4] => 1,
		[1, 2, 3, 5] => 1,
		[1, 2, 3, 6] => 1,
		[1, 2, 4, 5] => 4,
		[1, 2, 4, 6] => 4,
		[1, 2, 5, 6] => 5,
		[1, 3, 4, 5] => 5,
		[1, 3, 4, 6] => 6,
		[1, 3, 5, 6] => 5,
		[1, 4, 5, 6] => 5,
		[2, 3, 4, 5] => 5,
		[2, 3, 4, 6] => 6,
		[2, 3, 5, 6] => 6,
		[2, 4, 5, 6] => 6,
		[3, 4, 5, 6] => 6,
	)
	key = thirds_groups(group_tables)
	group = mapping[key]
	group_tables[group][3, :Team]
end

# ╔═╡ 265d31af-c6d0-4d97-8d31-27244c49485f
function opponent_1C(group_tables)
	mapping = Dict(
		[1, 2, 3, 4] => 4,
		[1, 2, 3, 5] => 5,
		[1, 2, 3, 6] => 6,
		[1, 2, 4, 5] => 5,
		[1, 2, 4, 6] => 6,
		[1, 2, 5, 6] => 6,
		[1, 3, 4, 5] => 4,
		[1, 3, 4, 6] => 4,
		[1, 3, 5, 6] => 6,
		[1, 4, 5, 6] => 6,
		[2, 3, 4, 5] => 4,
		[2, 3, 4, 6] => 4,
		[2, 3, 5, 6] => 5,
		[2, 4, 5, 6] => 5,
		[3, 4, 5, 6] => 5,
	)
	key = thirds_groups(group_tables)
	group = mapping[key]
	group_tables[group][3, :Team]
end

# ╔═╡ 14ade931-9ac0-4a6e-8952-acdeb4fcfecf
function opponent_1E(group_tables)
	mapping = Dict(
		[1, 2, 3, 4] => 2,
		[1, 2, 3, 5] => 2,
		[1, 2, 3, 6] => 2,
		[1, 2, 4, 5] => 1,
		[1, 2, 4, 6] => 1,
		[1, 2, 5, 6] => 2,
		[1, 3, 4, 5] => 3,
		[1, 3, 4, 6] => 3,
		[1, 3, 5, 6] => 3,
		[1, 4, 5, 6] => 4,
		[2, 3, 4, 5] => 2,
		[2, 3, 4, 6] => 3,
		[2, 3, 5, 6] => 3,
		[2, 4, 5, 6] => 4,
		[3, 4, 5, 6] => 4,
	)
	key = thirds_groups(group_tables)
	group = mapping[key]
	group_tables[group][3, :Team]
end

# ╔═╡ 0f2532e0-3235-49ba-bb43-f408393ead88
function opponent_1F(group_tables)
	mapping = Dict(
		[1, 2, 3, 4] => 3,
		[1, 2, 3, 5] => 3,
		[1, 2, 3, 6] => 3,
		[1, 2, 4, 5] => 2,
		[1, 2, 4, 6] => 2,
		[1, 2, 5, 6] => 1,
		[1, 3, 4, 5] => 1,
		[1, 3, 4, 6] => 1,
		[1, 3, 5, 6] => 1,
		[1, 4, 5, 6] => 1,
		[2, 3, 4, 5] => 3,
		[2, 3, 4, 6] => 2,
		[2, 3, 5, 6] => 2,
		[2, 4, 5, 6] => 2,
		[3, 4, 5, 6] => 3, 
	)
	key = thirds_groups(group_tables)
	group = mapping[key]
	group_tables[group][3, :Team]
end

# ╔═╡ 1b63b878-1b77-494f-8824-7908236ce805
function last_16_matches(tables)
	matches = [
		Match(tables[2][1, :Team], opponent_1B(tables), false),
		Match(tables[1][1, :Team], tables[3][2, :Team], false),
		Match(tables[6][1, :Team], opponent_1F(tables), false),
		Match(tables[4][2, :Team], tables[5][2, :Team], false),
		Match(tables[5][1, :Team], opponent_1E(tables), false),
		Match(tables[4][1, :Team], tables[6][2, :Team], false),
		Match(tables[3][1, :Team], opponent_1C(tables), false),
		Match(tables[1][2, :Team], tables[2][2, :Team], false),
	]
end

# ╔═╡ 845d6362-dc6f-44c4-bc96-492716b9d2e6
function play_tournament(groups)
	group_results = play_groupstage(groups)
	group_tables = [table(group_results[g_i, :]) for g_i in 1:size(group_results, 1)]
	last_16_results = [play_match(m) for m in last_16_matches(group_tables)]
	quarter_results = [play_match(m) for m in quarter_matches(last_16_results)]
	semi_results = [play_match(m) for m in semi_matches(quarter_results)]
	final_result = play_match(final_match(semi_results))
	TournamentResult(
		winner(final_result),
		Set(winner(r) for r in quarter_results),
		[t[1, :Team] for t in group_tables]
	)
end

# ╔═╡ 97804593-34c9-422b-8028-6737eb072a4e
play_tournament(groups)

# ╔═╡ c0e9655d-a596-4ba2-b6a6-f09b6f3459fe
function simulate_tournament(groups, n_samples)
	runs = [play_tournament(groups) for _ in 1:n_samples]
	best_winner = countmap(r.winner for r in runs) |> findmax |> reverse
	best_group_winners = [countmap(r.group_winners[g_i] for r in runs) |> findmax |> reverse for g_i in 1:6]
	semifinal_counts = countmap(r.semifinalists for r in runs)
	bsf = best_semifinal(semifinal_counts)
	
	BestTournamentResult(
		best_winner[1] => best_winner[2] / n_samples,
		bsf[1] => bsf[2],
		[bgw[1] => bgw[2] / n_samples for bgw in best_group_winners],
	)
end

# ╔═╡ 3f9184ed-b4fa-4c7f-ba9a-213ac0c4e67d
tournament_results = simulate_tournament(groups, 10_000)

# ╔═╡ Cell order:
# ╟─e6a2e856-ad88-4425-86ca-098359a0210c
# ╠═5d561ec1-3a9a-44dd-b1d9-dc2622e10b54
# ╟─da2b04dc-8177-48d7-b3cc-89d16444a8f2
# ╠═f17cab32-c91b-11eb-1d42-2de9769ace13
# ╠═5c24a3cf-9fd0-4aa3-8fa4-d415bdfcd588
# ╠═96f0a2a0-c5cb-4d04-9d38-47269fd58308
# ╠═d1682178-0dd6-413b-84e9-db90afd1b309
# ╟─1557b1cd-6596-426f-a1d2-e3d82a5a974d
# ╠═c1dee168-93bd-4640-b728-c54dfdb387a8
# ╠═e6dd90d9-80eb-4238-bf42-ce52d57f8b08
# ╟─b151cc55-a507-4f69-b548-555a2be3aba5
# ╠═6481fec5-04cf-4bf0-8330-3c7bffe129c3
# ╠═adc93b5b-da90-4e02-aad1-38d21690e432
# ╟─d4f9a8db-b796-47c5-b485-5720371d4460
# ╠═ce8e24a6-6fed-483b-95df-9b2cc42bda4f
# ╠═dc601549-1fd6-4fab-bbc6-6bc5f0305f68
# ╟─d434f991-2ed5-4d3e-bd5c-bf346ac8163d
# ╠═5e48f29f-cb09-4ca7-86c6-e17cc0278e90
# ╠═299f4b3c-b83e-4b8a-80c9-e41a96dda227
# ╠═f6c227bb-4f7e-4db1-9ebb-0bcab5977f9c
# ╠═294a097f-043e-4a2c-b7a3-d88177975ccd
# ╠═3cd0ccba-f12d-4d8d-a049-091916813816
# ╠═bf0cb9cf-ed9e-4cf5-8d14-cd8aa47d1ad9
# ╠═1b63b878-1b77-494f-8824-7908236ce805
# ╠═00c323a6-fa14-4ac3-b207-89ca48f7ff47
# ╠═5cd4fa9d-9d0a-434b-91a8-6b5c4f9ebe2b
# ╠═006dd1f9-2dea-4481-8951-ffa1bd21dd7a
# ╠═c7f66257-ba75-4815-a75b-59a9d616c532
# ╠═845d6362-dc6f-44c4-bc96-492716b9d2e6
# ╠═97804593-34c9-422b-8028-6737eb072a4e
# ╠═695793a4-af27-4cce-9d70-883d9c8d650d
# ╠═c0e9655d-a596-4ba2-b6a6-f09b6f3459fe
# ╟─de6ff9b0-97d9-46f0-878b-8c938cbea904
# ╠═d1f6466a-67ed-4b23-bd05-80db18792621
# ╟─05f595ea-18a4-4a3a-8efc-cbd8f00be497
# ╠═6c7fdb33-f5f9-4093-b529-9c5a3d1911a8
# ╟─1950efd1-1da8-4d8c-8403-5341a8c1d364
# ╠═40461daa-c6a3-4a5e-bd71-539db7fea679
# ╟─19a8ce2b-46d4-4a10-b422-46f8b6b07443
# ╠═b74b47bd-0ac9-4e04-8201-08375e39e55a
# ╟─664a812f-97f6-419a-bc1c-9e44a6c32456
# ╠═c2565dc1-7d05-44da-86e8-0eb020461403
# ╟─df720e43-ac9b-4dce-9614-e17ce81c2042
# ╠═09de1468-20f1-4cef-8fb5-07a989a6e163
# ╟─74298dfd-4ccb-45b2-977c-25452f3155f9
# ╠═3f9184ed-b4fa-4c7f-ba9a-213ac0c4e67d
# ╠═e927a22e-257e-41ba-bba3-d721cfa05b94
# ╠═61a11c9d-cddb-44c0-aaa1-619cc650e208
# ╠═265d31af-c6d0-4d97-8d31-27244c49485f
# ╠═14ade931-9ac0-4a6e-8952-acdeb4fcfecf
# ╠═0f2532e0-3235-49ba-bb43-f408393ead88
