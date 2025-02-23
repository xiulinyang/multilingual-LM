library('car')
# Mann-Whitney U test between Italian and shuffled Italian (local window size=2)
local2 = c(442.1773830486267,189.60268815394429,130.5610680720442,126.60738863235439,
           120.17472463448064,114.99450362130821,124.38615257514564,136.6004516362796,148.41961786422243,
           141.08022124677228)
control = c(212.83515999525508,185.2305620465523,162.73584696686027,129.57583614306114,
            121.16810972203123,119.78409492516397,117.68279494579467,119.43095620596318,120.24627744925134,
            118.40217540442217)

data <- data.frame(
  value = c(local2, control),
  group = rep(c("local2", "control"), each = 10)
)

# Perform Levene's test
leveneTest(value ~ group, data = data)

shapiro.test(local2)
shapiro.test(control)

wilcox.test(local2, control, alternative = "two.sided")

# Spearman's rho test between tokens per word (TPW) and perplexity
token <-c(2.19, 2.049, 2.047, 1.98, 1.61, 1.40, 1.68, 1.51, 1.81, 1.45, 1.67)
ppl <-c(364.4551576,263.21880266,172.35574443,168.66187223,96.61820174,118.53043073,
        93.61212663,112.3661769,56.6383911,76.41691815,58.33768428)

shapiro.test(token)
shapiro.test(ppl)
result_spearman <- cor.test(token, ppl, method = "spearman")
result_spearman



