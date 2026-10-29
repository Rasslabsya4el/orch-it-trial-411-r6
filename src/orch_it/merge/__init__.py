"""The merge flow (#400): one open pull request per ticket at a time, bound to its head (`pr open`), the merge line
that lands green pull requests one by one under a repo lock (`merge line`), the ticket following its
pull request (`follow`), main checked after every merge (`main check`) and the scheduled sweep
(`reconcile`)."""
