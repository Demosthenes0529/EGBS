(set-logic QF_BV)


(declare-const a (_ BitVec 4))
(declare-const b (_ BitVec 4))
(declare-const c (_ BitVec 4))

(define-fun ZERO () (_ BitVec 4) (_ bv0 4))

(assert (= a (_ bv1 4)))
(assert (= b (_ bv1 4)))
(assert (= c (_ bv1 4))) 


(define-fun a1 () (_ BitVec 4) (bvshl a (_ bv1 4)))
(define-fun b1 () (_ BitVec 4) (bvshl b (_ bv1 4)))
(define-fun c1 () (_ BitVec 4) (bvshl c (_ bv1 4)))

(define-fun not_a1 () (_ BitVec 4) (bvnot a1))

(define-fun eq1 () (_ BitVec 4) 
  (bvand (bvand (bvxor not_a1 b1) (bvxor not_a1 c1))))

(define-fun xor_sum () (_ BitVec 4)
  (bvxor a (bvxor b (bvxor c b1))))

(define-fun condition () (_ BitVec 4)
  (bvand eq1 xor_sum))

(assert (= condition ZERO))

(check-sat)
(get-value (a b c condition))