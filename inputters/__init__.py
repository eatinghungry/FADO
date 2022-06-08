
from inputters.strat import Inputter as strat
from inputters.startrl import Inputter as stratrl
from inputters.vanilla import Inputter as vanilla
from inputters.strat_dqn import Inputter as stratdqn
from inputters.strat_prompt import Inputter as strat_prompt


inputters = {
    'vanilla': vanilla,
    'strat_prompt': strat_prompt,
    'strat': strat,
    'stratrl': stratrl,
    'stratdqn': stratdqn
}



