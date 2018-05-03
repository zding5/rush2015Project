import _dynet as dy

from RNNutils import dy_softplus, dy_log

class BiGRU:
    def __init__(self, model, emb_dim, hid_dim):
        pc = model.add_subcollection()

        # BiGRU
        self.BiGRUBuilder = dy.BiRNNBuilder(1, emb_dim, hid_dim, pc, dy.GRUBuilder)

        self.pc = pc
        self.spec = (emb_dim, hid_dim)

    def __call__(self, x):
        return self.BiGRUBuilder.transduce(x)

    def associate_parameters(self):
        pass

    @staticmethod
    def from_spec(spec, model):
        emb_dim, hid_dim = spec
        return BiGRU(model, emb_dim, hid_dim)

    def param_collection(self):
        return self.pc

class RecurrentGenerativeDecoder:
    def __init__(self, model, emb_dim, hid_dim, lat_dim, out_dim):
        pc = model.add_subcollection()

        # First and Second GRUs
        self.firstGRUBuilder  = dy.GRUBuilder(1, emb_dim, hid_dim, pc)
        self.secondGRUBuilder = dy.GRUBuilder(1, emb_dim+hid_dim, hid_dim, pc)

        # Attention layer
        self._Wdhh = pc.add_parameters((hid_dim, hid_dim))
        self._Wehh = pc.add_parameters((hid_dim, hid_dim))
        self._ba   = pc.add_parameters((hid_dim), init=dy.ConstInitializer(0))
        self._v    = pc.add_parameters((hid_dim))

        # decoder
        self._Wdyzh = pc.add_parameters((hid_dim, lat_dim))
        self._Wdzhh = pc.add_parameters((hid_dim, hid_dim))
        self._bdyh  = pc.add_parameters((hid_dim), init=dy.ConstInitializer(0))

        # Output layer
        self._Wdhy = pc.add_parameters((out_dim, hid_dim))
        self._bdhy = pc.add_parameters((out_dim), init=dy.ConstInitializer(0))

        # Initial state
        self._z_0 = pc.add_parameters((lat_dim))

        self.lat_dim = lat_dim
        self.pc = pc
        self.spec = (emb_dim, hid_dim, lat_dim, out_dim)

    def __call__(self, t, tm1s=None, test=False):
        if test:
            t_tm1   = t
            hd1_tm1 = tm1s[0]
            hd2_tm1 = tm1s[1]
            # z_tm1   = tm1s[2]

            # First GRU
            hd1_t = self.firstGRUBuilder.initial_state().set_s([hd1_tm1]).add_input(t_tm1).output()

            # Attention layer
            e_t = dy.concatenate([dy.dot_product(self.v, dy.tanh(self.Wdhh*hd1_t + Wehh_he_j + self.ba)) for Wehh_he_j in self.Wehh_he])
            a_t = dy.softmax(e_t)
            c_t = dy.esum([dy.cmult(a_tj, he_j) for a_tj, he_j in zip(a_t, self.he)])

            # Second GRU
            hd2_t = self.secondGRUBuilder.initial_state().set_s([hd2_tm1]).add_input(dy.concatenate([c_t, t_tm1])).output()

            # decode
            hdy_t = dy.tanh(self.Wdzhh*hd2_t + self.bdyh)

            # Output layer with softmax
            y_t = dy.softmax(self.Wdhy*hdy_t + self.bdhy)

            return y_t, hd1_t, hd2_t

        else:
            # First GRU
            hd1 = self.firstGRUBuilder.initial_state([self.hd1_0]).transduce(t)

            # Attention layer
            c = [] # context vectors
            for i, hd1_t in enumerate(hd1):
                e_t = dy.concatenate([dy.dot_product(self.v, dy.tanh(self.Wdhh*hd1_t + Wehh_he_j + self.ba)) for Wehh_he_j in self.Wehh_he])
                a_t = dy.softmax(e_t)
                c_t = dy.esum([dy.cmult(a_tj, he_j) for a_tj, he_j in zip(a_t, self.he)])
                c.append(c_t)

            # print(c)
            # Second GRU
            hd2_input = [dy.concatenate([c_t, t_tm1]) for c_t, t_tm1 in zip(c, t)]
            hd2 = self.secondGRUBuilder.initial_state([self.hd2_0]).transduce(hd2_input)

            hd1_ = [self.hd1_0] + hd1[:-1] # [hd1_0, hd1_1, ..., hd1_Tm1]
            KL = []
            y = []
            for i, (t_tm1, hd1_tm1, hd2_t) in enumerate(zip(t, hd1_, hd2)):

                # decode
                # hdy_t = dy.tanh(self.Wdyzh*z_t + self.Wdzhh*hd2_t + self.bdyh)
                hdy_t = dy.tanh(self.Wdzhh*hd2_t + self.bdyh)

                # Output layer without softmax
                y_t = self.Wdhy*hdy_t + self.bdhy
                y.append(y_t)


            # return y, KL
            return y

    def associate_parameters(self):
        self.Wdhh  = dy.parameter(self._Wdhh)
        self.Wehh  = dy.parameter(self._Wehh)
        self.ba    = dy.parameter(self._ba)
        self.v     = dy.parameter(self._v)
        self.Wdyzh = dy.parameter(self._Wdyzh)
        self.Wdzhh = dy.parameter(self._Wdzhh)
        self.bdyh  = dy.parameter(self._bdyh)
        self.Wdhy  = dy.parameter(self._Wdhy)
        self.bdhy  = dy.parameter(self._bdhy)

    def set_initial_states(self, he):
        hd_0 = dy.average(he)
        self.he = he
        self.hd1_0 = hd_0
        self.hd2_0 = hd_0
        self.Wehh_he = [self.Wehh*he_j for he_j in he]

    @staticmethod
    def from_spec(spec, model):
        emb_dim, hid_dim, lat_dim, out_dim = spec
        return RecurrentGenerativeDecoder(model, emb_dim, hid_dim, lat_dim, out_dim)

    def param_collection(self):
        return self.pc
