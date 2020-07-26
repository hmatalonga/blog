<template>
  <div>
    <h1
      class="text-5xl font-semibold leading-tight text-gray-800 md:text-6xl lg:text-7xl lg:font-medium xl:text-8xl"
    >
      Blog
    </h1>
    <div class="mt-8 md:mt-10 xl:mt-16"></div>
    <ul class="list-none leading-snug space-y-8">
      <li
        v-for="post of posts"
        :key="post.slug"
        class="mt-1 text-lg text-gray-800 font-semibold md:text-2xl"
      >
        <nuxt-link :to="post.path">{{ post.title }}</nuxt-link>
      </li>
    </ul>
  </div></template
>

<script>
export default {
  async asyncData({ $content, params }) {
    const posts = await $content('blog')
      .where({ published: true })
      .sortBy('date', 'desc')
      .fetch()

    return {
      posts,
    }
  },
  head() {
    return {
      title: 'Blog',
    }
  },
}
</script>
