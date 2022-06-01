
from inputters.strat import Inputter as strat
from inputters.startrl import Inputter as stratrl
from inputters.vanilla import Inputter as vanilla
from inputters.strat_dqn import Inputter as stratdqn


inputters = {
    'vanilla': vanilla,
    'strat': strat,
    'stratrl': stratrl,
    'stratdqn': stratdqn
}



