def arg_parser(args,key,default_val=None,required=False):
    """
    argsのなかにkeyがあるならその値をないならdefaultの値をreturnする。

    default_val: if not specified, None is set
    """
    if key in args.keys():
        return args[key]
    else:
        assert required==False,"arg ::{}:: is requiread".format(key)
        return default_val
