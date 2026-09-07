import aggregate_t4_m30_experiment_a_r5 as impl
from t4_m30_experiment_a_r7_authorization import require_claim,RECEIPT
impl.require_claim=require_claim;impl.RECEIPT=RECEIPT
main=impl.main
if __name__=='__main__':main()
