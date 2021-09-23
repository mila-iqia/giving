
giving.obs
==========

.. automodule:: giving.obs

    .. autoclass:: ObservableProxy

        .. automethod:: accum
        .. automethod:: breakpoint
        .. automethod:: breakword
        .. automethod:: display
        .. automethod:: give
        .. automethod:: ksubscribe
        .. automethod:: kwrap
        .. automethod:: pipe(*ops)
        .. *
        .. automethod:: print
        .. automethod:: subscribe(observer=None, on_next=None, on_error=None, on_completed=None)
        .. automethod:: wrap
        .. automethod:: __or__
        .. automethod:: __rshift__
        .. automethod:: __getitem__

        The methods below are shortcuts to the corresponding operators in :mod:`giving.operators`,
        applied to ``self``:

        .. function:: affix(...)

            See :func:`~giving.operators.affix`

        .. function:: all(...)

            See :func:`~giving.operators.all`

        .. function:: amb(...)

            See :func:`~giving.operators.amb`

        .. function:: as_(...)

            See :func:`~giving.operators.as_`

        .. function:: as_observable(...)

            See :func:`~giving.operators.as_observable`

        .. function:: augment(...)

            See :func:`~giving.operators.augment`

        .. function:: average(...)

            See :func:`~giving.operators.average`

        .. function:: average_and_variance(...)

            See :func:`~giving.operators.average_and_variance`

        .. function:: buffer(...)

            See :func:`~giving.operators.buffer`

        .. function:: buffer_toggle(...)

            See :func:`~giving.operators.buffer_toggle`

        .. function:: buffer_when(...)

            See :func:`~giving.operators.buffer_when`

        .. function:: buffer_with_count(...)

            See :func:`~giving.operators.buffer_with_count`

        .. function:: buffer_with_time(...)

            See :func:`~giving.operators.buffer_with_time`

        .. function:: buffer_with_time_or_count(...)

            See :func:`~giving.operators.buffer_with_time_or_count`

        .. function:: catch(...)

            See :func:`~giving.operators.catch`

        .. function:: collect_between(...)

            See :func:`~giving.operators.collect_between`

        .. function:: combine_latest(...)

            See :func:`~giving.operators.combine_latest`

        .. function:: concat(...)

            See :func:`~giving.operators.concat`

        .. function:: contains(...)

            See :func:`~giving.operators.contains`

        .. function:: count(...)

            See :func:`~giving.operators.count`

        .. function:: debounce(...)

            See :func:`~giving.operators.debounce`

        .. function:: default_if_empty(...)

            See :func:`~giving.operators.default_if_empty`

        .. function:: delay(...)

            See :func:`~giving.operators.delay`

        .. function:: delay_subscription(...)

            See :func:`~giving.operators.delay_subscription`

        .. function:: delay_with_mapper(...)

            See :func:`~giving.operators.delay_with_mapper`

        .. function:: dematerialize(...)

            See :func:`~giving.operators.dematerialize`

        .. function:: distinct(...)

            See :func:`~giving.operators.distinct`

        .. function:: distinct_until_changed(...)

            See :func:`~giving.operators.distinct_until_changed`

        .. function:: do(...)

            See :func:`~giving.operators.do`

        .. function:: do_action(...)

            See :func:`~giving.operators.do_action`

        .. function:: do_while(...)

            See :func:`~giving.operators.do_while`

        .. function:: element_at(...)

            See :func:`~giving.operators.element_at`

        .. function:: element_at_or_default(...)

            See :func:`~giving.operators.element_at_or_default`

        .. function:: exclusive(...)

            See :func:`~giving.operators.exclusive`

        .. function:: expand(...)

            See :func:`~giving.operators.expand`

        .. function:: filter(...)

            See :func:`~giving.operators.filter`

        .. function:: filter_indexed(...)

            See :func:`~giving.operators.filter_indexed`

        .. function:: finally_action(...)

            See :func:`~giving.operators.finally_action`

        .. function:: find(...)

            See :func:`~giving.operators.find`

        .. function:: find_index(...)

            See :func:`~giving.operators.find_index`

        .. function:: first(...)

            See :func:`~giving.operators.first`

        .. function:: first_or_default(...)

            See :func:`~giving.operators.first_or_default`

        .. function:: flat_map(...)

            See :func:`~giving.operators.flat_map`

        .. function:: flat_map_indexed(...)

            See :func:`~giving.operators.flat_map_indexed`

        .. function:: flat_map_latest(...)

            See :func:`~giving.operators.flat_map_latest`

        .. function:: fork_join(...)

            See :func:`~giving.operators.fork_join`

        .. function:: format(...)

            See :func:`~giving.operators.format`

        .. function:: getitem(...)

            See :func:`~giving.operators.getitem`

        .. function:: group_by(...)

            See :func:`~giving.operators.group_by`

        .. function:: group_by_until(...)

            See :func:`~giving.operators.group_by_until`

        .. function:: group_join(...)

            See :func:`~giving.operators.group_join`

        .. function:: group_wrap(...)

            See :func:`~giving.operators.group_wrap`

        .. function:: ignore_elements(...)

            See :func:`~giving.operators.ignore_elements`

        .. function:: is_empty(...)

            See :func:`~giving.operators.is_empty`

        .. function:: join(...)

            See :func:`~giving.operators.join`

        .. function:: keep(...)

            See :func:`~giving.operators.keep`

        .. function:: kfilter(...)

            See :func:`~giving.operators.kfilter`

        .. function:: kmap(...)

            See :func:`~giving.operators.kmap`

        .. function:: kmerge(...)

            See :func:`~giving.operators.kmerge`

        .. function:: last(...)

            See :func:`~giving.operators.last`

        .. function:: last_or_default(...)

            See :func:`~giving.operators.last_or_default`

        .. function:: map(...)

            See :func:`~giving.operators.map`

        .. function:: map_indexed(...)

            See :func:`~giving.operators.map_indexed`

        .. function:: materialize(...)

            See :func:`~giving.operators.materialize`

        .. function:: max(...)

            See :func:`~giving.operators.max`

        .. function:: merge(...)

            See :func:`~giving.operators.merge`

        .. function:: merge_all(...)

            See :func:`~giving.operators.merge_all`

        .. function:: min(...)

            See :func:`~giving.operators.min`

        .. function:: multicast(...)

            See :func:`~giving.operators.multicast`

        .. function:: observe_on(...)

            See :func:`~giving.operators.observe_on`

        .. function:: on_error_resume_next(...)

            See :func:`~giving.operators.on_error_resume_next`

        .. function:: pairwise(...)

            See :func:`~giving.operators.pairwise`

        .. function:: partition(...)

            See :func:`~giving.operators.partition`

        .. function:: partition_indexed(...)

            See :func:`~giving.operators.partition_indexed`

        .. function:: pluck(...)

            See :func:`~giving.operators.pluck`

        .. function:: pluck_attr(...)

            See :func:`~giving.operators.pluck_attr`

        .. function:: publish(...)

            See :func:`~giving.operators.publish`

        .. function:: publish_value(...)

            See :func:`~giving.operators.publish_value`

        .. function:: reduce(...)

            See :func:`~giving.operators.reduce`

        .. function:: ref_count(...)

            See :func:`~giving.operators.ref_count`

        .. function:: repeat(...)

            See :func:`~giving.operators.repeat`

        .. function:: replay(...)

            See :func:`~giving.operators.replay`

        .. function:: retry(...)

            See :func:`~giving.operators.retry`

        .. function:: roll(...)

            See :func:`~giving.operators.roll`

        .. function:: sample(...)

            See :func:`~giving.operators.sample`

        .. function:: scan(...)

            See :func:`~giving.operators.scan`

        .. function:: sequence_equal(...)

            See :func:`~giving.operators.sequence_equal`

        .. function:: share(...)

            See :func:`~giving.operators.share`

        .. function:: single(...)

            See :func:`~giving.operators.single`

        .. function:: single_or_default(...)

            See :func:`~giving.operators.single_or_default`

        .. function:: single_or_default_async(...)

            See :func:`~giving.operators.single_or_default_async`

        .. function:: skip(...)

            See :func:`~giving.operators.skip`

        .. function:: skip_last(...)

            See :func:`~giving.operators.skip_last`

        .. function:: skip_last_with_time(...)

            See :func:`~giving.operators.skip_last_with_time`

        .. function:: skip_until(...)

            See :func:`~giving.operators.skip_until`

        .. function:: skip_until_with_time(...)

            See :func:`~giving.operators.skip_until_with_time`

        .. function:: skip_while(...)

            See :func:`~giving.operators.skip_while`

        .. function:: skip_while_indexed(...)

            See :func:`~giving.operators.skip_while_indexed`

        .. function:: skip_with_time(...)

            See :func:`~giving.operators.skip_with_time`

        .. function:: slice(...)

            See :func:`~giving.operators.slice`

        .. function:: some(...)

            See :func:`~giving.operators.some`

        .. function:: sole(...)

            See :func:`~giving.operators.sole`

        .. function:: starmap(...)

            See :func:`~giving.operators.starmap`

        .. function:: starmap_indexed(...)

            See :func:`~giving.operators.starmap_indexed`

        .. function:: start_with(...)

            See :func:`~giving.operators.start_with`

        .. function:: subscribe_on(...)

            See :func:`~giving.operators.subscribe_on`

        .. function:: sum(...)

            See :func:`~giving.operators.sum`

        .. function:: switch_latest(...)

            See :func:`~giving.operators.switch_latest`

        .. function:: tag(...)

            See :func:`~giving.operators.tag`

        .. function:: take(...)

            See :func:`~giving.operators.take`

        .. function:: take_last(...)

            See :func:`~giving.operators.take_last`

        .. function:: take_last_buffer(...)

            See :func:`~giving.operators.take_last_buffer`

        .. function:: take_last_with_time(...)

            See :func:`~giving.operators.take_last_with_time`

        .. function:: take_until(...)

            See :func:`~giving.operators.take_until`

        .. function:: take_until_with_time(...)

            See :func:`~giving.operators.take_until_with_time`

        .. function:: take_while(...)

            See :func:`~giving.operators.take_while`

        .. function:: take_while_indexed(...)

            See :func:`~giving.operators.take_while_indexed`

        .. function:: take_with_time(...)

            See :func:`~giving.operators.take_with_time`

        .. function:: throttle(...)

            See :func:`~giving.operators.throttle`

        .. function:: throttle_first(...)

            See :func:`~giving.operators.throttle_first`

        .. function:: throttle_with_mapper(...)

            See :func:`~giving.operators.throttle_with_mapper`

        .. function:: throttle_with_timeout(...)

            See :func:`~giving.operators.throttle_with_timeout`

        .. function:: time_interval(...)

            See :func:`~giving.operators.time_interval`

        .. function:: timeout(...)

            See :func:`~giving.operators.timeout`

        .. function:: timeout_with_mapper(...)

            See :func:`~giving.operators.timeout_with_mapper`

        .. function:: timestamp(...)

            See :func:`~giving.operators.timestamp`

        .. function:: to_dict(...)

            See :func:`~giving.operators.to_dict`

        .. function:: to_future(...)

            See :func:`~giving.operators.to_future`

        .. function:: to_iterable(...)

            See :func:`~giving.operators.to_iterable`

        .. function:: to_list(...)

            See :func:`~giving.operators.to_list`

        .. function:: to_marbles(...)

            See :func:`~giving.operators.to_marbles`

        .. function:: to_set(...)

            See :func:`~giving.operators.to_set`

        .. function:: variance(...)

            See :func:`~giving.operators.variance`

        .. function:: where(...)

            See :func:`~giving.operators.where`

        .. function:: where_any(...)

            See :func:`~giving.operators.where_any`

        .. function:: while_do(...)

            See :func:`~giving.operators.while_do`

        .. function:: window(...)

            See :func:`~giving.operators.window`

        .. function:: window_toggle(...)

            See :func:`~giving.operators.window_toggle`

        .. function:: window_when(...)

            See :func:`~giving.operators.window_when`

        .. function:: window_with_count(...)

            See :func:`~giving.operators.window_with_count`

        .. function:: window_with_time(...)

            See :func:`~giving.operators.window_with_time`

        .. function:: window_with_time_or_count(...)

            See :func:`~giving.operators.window_with_time_or_count`

        .. function:: with_latest_from(...)

            See :func:`~giving.operators.with_latest_from`

        .. function:: zip(...)

            See :func:`~giving.operators.zip`

        .. function:: zip_with_iterable(...)

            See :func:`~giving.operators.zip_with_iterable`

        .. function:: zip_with_list(...)

            See :func:`~giving.operators.zip_with_list`

