# How to divide profits

## goals

- nobody thinks it's significantly unfair
- be simple enough
- be sustainable for the future profits
- motivate helping each other

## procedure

### terms

- profits = revenue - costs
- n = number of workers

### suggestion

- poll
  - n / 2 polls per person
  - one cannot poll to oneself
  - n random polls are appended to hide poll
  - abstention or polls left are treated as random polls instead

```py
MAX_WORK_HOURS_PER_DAY = 8
MAX_WORK_DAYS_PER_WEEK = 5
NUM_POLL_PER_PERSON = 10
MAX_POLL_TO_GIVE_TO_A_PERSON = 3
N_SMOOTHING_POLLS_TO_A_PERSON = 2
N_RANDOM_POLLS = n

NATIONAL_MIN_WAGE_PER_HOUR = 8590 # in 2020
MIN_INCOME_PER_HOUR = max(NATIONAL_MIN_WAGE_PER_HOUR * 2, 50000000 / ((52 * 5 - 20)*8)
P_REINVESTMENT = 0.3

def get_income(revenue, costs, n, actual_polls, work_hours):
    income = [0] * n
    left = revenue
    left -= costs

    sum_work_hours = sum(work_hours)

    # put aside minimum income
    total_min_wage_required = sum_work_hours * MIN_INCOME_PER_HOUR

    if left < total_min_wage:
        for i in range(n):
            incomes[i] += floor(left * work_hours[i] / sum_work_hours)
    else:
        for i in range(n):
            incomes[i] += floor(total_min_wage_required * work_hours[i] / sum_work_hours)
        incomes += minimum_income
        left -= total_min_wage_required

    # reinvestment
    left -= floor(max(P_REINVESTMENT * left, 0))

    # incentive
    polls = list(actual_polls)
    for i in range(n):
        polls[i] += N_SMOOTHING_POLLS_TO_A_PERSON
    for _ in range(n):
        i = n * random.randrange(n)
        polls[i] += 1

    sum_polls = sum(polls)

    for i in range(n):
        incomes[i] += N_SMOOTHING_POLLS_TO_A_PERSON

    reinvestment = left

    return incomes, reinvestment
```

## tips

- make a document
