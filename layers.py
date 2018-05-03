import _dynet as dy
import math
class ABS:
    def __init__(self, model, emb_dim, hid_dim, vocab_size, q, c,  context , input_dim , output_dim , encoder_type='attention',  full_context=True):
        pc = model.add_subcollection()

        # Neural language model
        self.E  = pc.add_lookup_parameters((vocab_size, emb_dim))
        self._U = pc.add_parameters((hid_dim, c*emb_dim))
        self._V = pc.add_parameters((vocab_size, hid_dim))
        self.c  = c

        # Encoder
        self.F            = pc.add_lookup_parameters((vocab_size, hid_dim))
        self._W           = pc.add_parameters((vocab_size, hid_dim))
        self.encoder_type = encoder_type
        self.q            = q

        # Attention-based encoder
        if self.encoder_type == 'attention':
            self.G  = pc.add_lookup_parameters((vocab_size, emb_dim))
            self._P = pc.add_parameters((hid_dim, c*emb_dim))



        # Cov_Attention-based encoder
        elif self.encoder_type == 'Conv_attention':
            self.G  = pc.add_lookup_parameters((vocab_size, emb_dim))
            self._P = pc.add_parameters((hid_dim, c*emb_dim))

            # TDNN Convolution part

            self.input_dim = input_dim
            self.output_dim = hid_dim #output_dim
            self.check_valid_context(context)
            self.kernel_width, self.context = self.get_kernel_width(context,full_context)
            self.full_context = full_context
            stdv = 1./math.sqrt(input_dim)
            #self.kernel = nn.Parameter(torch.Tensor(output_dim, input_dim, self.kernel_width).normal_(0,stdv))
            #normal_(0,stdv) fills the input Tensor with values drawn from the normal
            self._kernel = pc.add_parameters(( 1, self.kernel_width, output_dim, input_dim), init='normal', mean=0.0, std=stdv)
            #self.bias = nn.Parameter(torch.Tensor(output_dim).normal_(0,stdv))
            self._bias = pc.add_parameters(output_dim, init='normal', mean=0.0, std=stdv)

        self.pc = pc
        self.spec = (emb_dim, hid_dim, vocab_size, q, c, encoder_type, context, input_dim, output_dim, full_context)

    def __call__(self, x=None, t=None, test=False):
        if test:
            tt_embs = [dy.lookup(self.E, t_t) for t_t in t]

            if self.encoder_type == 'bow':
                # Neural language model
                tt_c    = dy.concatenate(tt_embs)
                h       = dy.tanh(self.U*tt_c)

                # Output with softmax
                y_t     = dy.softmax(self.V*h + self.W_enc)

            elif self.encoder_type == 'attention':
                ttp_embs = [dy.lookup(self.G, t_t) for t_t in t]

                # Neural language model
                tt_c = dy.concatenate(tt_embs)
                h    = dy.tanh(self.U*tt_c)

                # Attention
                ttp_c = dy.concatenate(ttp_embs)
                p     = dy.softmax(self.xt*self.P*ttp_c) # Attention weight
                enc   = self.xb*p                        # Context vector

                # Output with softmax
                y_t = dy.softmax(self.V*h + self.W*enc)

            return y_t


        else:
            xt_embs = [dy.lookup(self.F, x_t) for x_t in x] #F=(vocab_size, hid_dim), xt_embs = len(x)*[ (hid_dim)]
            tt_embs = [dy.lookup(self.E, t_t) for t_t in t] #E=(vocab_size, emb_dim),  tt_embs = len(t)*[ (emb_dim)]

            y = []
            if self.encoder_type == 'bow':
                # BoW
                enc = dy.average(xt_embs) #(hid_dim)
                W_enc = self.W*enc #(vocab_size, hid_dim)*(hid_dim)
                for i in range(len(t)-self.c+1):
                    # Neural language model
                    tt_c = dy.concatenate(tt_embs[i:i+self.c]) #(c* emb_dim)
                    h = dy.tanh(self.U*tt_c) #(hid_dim, c*emb_dim)*(c* emb_dim)

                    # Output without softmax
                    y_t = self.V*h + W_enc # (vocab_size, hid_dim)*(hid_dim) + (vocab_size)
                    y.append(y_t) #(vocab_size)
 
            elif self.encoder_type == 'attention':
                xb = dy.concatenate([dy.esum(xt_embs[max(i-self.q,0):min(len(x)-1+1,i+self.q+1)])/self.q for i in range(len(x))], d=1)
                #(hid_dim, len(x)) 
                xt = dy.transpose(dy.concatenate(xt_embs, d=1)) 
                # (len(x), hid_dim)
                ttp_embs = [dy.lookup(self.G, t_t) for t_t in t]
                # len(t)*[( emb_dim)]
                print (ttp_embs.dim())
                for i in range(len(t)-self.c+1):
                    # Neural language model
                    tt_c = dy.concatenate(tt_embs[i:i+self.c]) #(c* emb_dim)
                    h = dy.tanh(self.U*tt_c) #(hid_dim, c*emb_dim)*(c* emb_dim)

                    # Attention
                    ttp_c = dy.concatenate(ttp_embs[i:i+self.c]) # Window-sized embedding
                    #(c*emb_dim)
                    p     = dy.softmax(xt*self.P*ttp_c) # Attention weight
                    #(len(x), hid_dim)*(hid_dim, c*emb_dim)*(c*emb_dim)
                    enc = xb*p # Context vector
                    #(hid_dim, len(x)) *(len(x))

                    # Output without softmax
                    y_t = self.V*h + self.W*enc  #(vocab_size, hid_dim)*(hid_dim) + (vocab_size, hid_dim)*(hid_dim)
                    y.append(y_t)

            elif self.encoder_type == 'Conv_attention':
                xb = dy.concatenate([dy.esum(xt_embs[max(i-self.q,0):min(len(x)-1+1,i+self.q+1)])/self.q for i in range(len(x))], d=1)
                #(hid_dim, len(x)) 
                xt = dy.transpose(dy.concatenate(xt_embs, d=1)) 
                print ('xt.dim() = ', xt.dim())
                conv_out = dy.rectify(self.special_convolution(xt, self.kernel, self.context, self.bias))
                print ('conv_out : ',  conv_out.dim() )
                # (len(x), hid_dim)
                ttp_embs = [dy.lookup(self.G, t_t) for t_t in t]
                # len(t)*[( emb_dim)]
                #print ('ttp_embs[0].dim() = ', ttp_embs[0].dim(), len(ttp_embs) )
                for i in range(len(t)-self.c+1):
                    # Neural language model
                    tt_c = dy.concatenate(tt_embs[i:i+self.c]) #(c* emb_dim)
                    h = dy.tanh(self.U*tt_c) #(hid_dim, c*emb_dim)*(c* emb_dim)

                    # Attention
                    ttp_c = dy.concatenate(ttp_embs[i:i+self.c]) # Window-sized embedding

                    #(c*emb_dim)

                    p     = dy.softmax(conv_out*self.P*ttp_c) # Attention weight
                    #p     = dy.softmax(xt*self.P*ttp_c) # Attention weight
                    #(len(x), hid_dim)*(hid_dim, c*emb_dim)*(c*emb_dim)
                    enc = xb*p # Context vector
                    #(hid_dim, len(x)) *(len(x))

                    # Output without softmax
                    y_t = self.V*h + self.W*enc  #(vocab_size, hid_dim)*(hid_dim) + (vocab_size, hid_dim)*(hid_dim)
                    y.append(y_t)

            return y

    def special_convolution(self, x, kernel, context, bias):
        """
        This function performs the weight multiplication given an arbitrary context. Cannot directly use convolution because in case of only particular frames of context,
        one needs to select only those frames and perform a convolution across all batch items and all output dimensions of the kernel.
        """
        input_size = x.dim() 
        print('x.dim() = ', x.dim() )
        assert len(input_size) == 2, 'Input tensor dimensionality is incorrect. Should be a 3D tensor'
        [[ input_sequence_length, input_dim], batch_size] = input_size
        #x = dy.transpose(x,dims = [1,0])#.transpose(1,2)
        print('x.dim() = ', x.dim() )
        # Allocate memory for output
        valid_steps = self.get_valid_steps(self.context, input_sequence_length)
        #xs = Variable(self.bias.data.new(batch_size, kernel.size()[0], len(valid_steps)))
        print( 'kernel.dim()  = ', kernel.dim() )
        #xs = dy.ones(( len(valid_steps, kernel.dim()[0][0] )), batch_size=batch_size)
        #print( 'xs = ', xs.dim())

        # Perform the convolution with relevant input frames
        pt=[]
        for c, i in enumerate(valid_steps):
            #features = torch.index_select(x, 2, context+i)
            features = dy.strided_select(x, [1,1],[i,0], [context[-1]+i+1,input_dim]  )
            [[i,s],b]= features.dim()
            features = dy.reshape(features, (1, i, s))
            # Do the convolution
            #xs[:, :, c] = F.conv1d(features, kernel, bias=bias)[:, :, 0]
            #print ('features :',features.dim())
            #print ('kernel :', kernel.dim())
            #import pdb; pdb.set_trace()

            uu = dy.conv2d(features, kernel, [1, 1], is_valid=True)
            #print (uu)
            rr=  dy.pick(uu, index=0, dim=0) 
            pt.append(rr)		
        xs = dy.concatenate(pt[:1]+pt+pt[-1:], d=0)
        return xs


    def set_initial_states(self, x):
        self.xt_embs = [dy.lookup(self.F, x_t) for x_t in x]

        if self.encoder_type == 'bow':
            self.W_enc = self.W*dy.average(self.xt_embs)

        elif self.encoder_type == 'attention':
            self.xb       = dy.concatenate(
                [dy.esum(self.xt_embs[max(i-self.q,0):min(len(x)-1+1,i+self.q+1)])/self.q for i in range(len(x))],
                d=1
            )
            self.xt       = dy.transpose(dy.concatenate(self.xt_embs, d=1))

    def associate_parameters(self):
        self.U = dy.parameter(self._U)
        self.V = dy.parameter(self._V)
        self.W = dy.parameter(self._W)

        if self.encoder_type == 'attention':
            self.P = dy.parameter(self._P)
        elif self.encoder_type == 'Conv_attention':
            self.P = dy.parameter(self._P)
            self.kernel = dy.parameter(self._kernel)
            self.bias = dy.parameter(self._bias)

    @staticmethod
    def from_spec(spec, model):
        emb_dim, hid_dim, vocab_size, q, c, encoder_type, context, input_dim, output_dim, full_context = spec
        return ABS(emb_dim, hid_dim, vocab_size, q, c, encoder_type, context, input_dim, output_dim, full_context)

    def param_collection(self):
        return self.pc

    @staticmethod
    def check_valid_context(context):
        # here context is still a list
        assert context[0] <= context[-1], 'Input tensor dimensionality is incorrect. Should be a 3D tensor'

    @staticmethod
    def get_kernel_width(context, full_context):
        if full_context:
            context = range(context[0],context[-1]+1)
        return len(context), context

    @staticmethod
    def get_valid_steps(context, input_sequence_length):
        start = 0 if context[0] >= 0 else -1*context[0]
        end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
        return range(start, end)
