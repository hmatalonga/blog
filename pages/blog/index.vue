<template>
  <div>
    <h1
      class="text-5xl font-semibold leading-tight text-gray-800 md:text-6xl lg:text-7xl lg:font-medium xl:text-8xl"
    >
      Blog
    </h1>
    <div class="mt-8 md:mt-10 xl:mt-16"></div>
    <h2 class="mt-6 text-lg text-gray-700 md:text-xl lg:text-2xl">
      This is where you can find my writings. If you like my content, please
      consider
      <a
        class="text-indigo-600 underline"
        href="https://www.buymeacoffee.com/hmatalonga"
        target="_blank"
        >buying me a coffee</a
      >.
    </h2>
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
  </div>
</template>

<script>
export default {
  async asyncData({ $content }) {
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
      link: [
        {
          rel: 'canonical',
          href: `https://hmatalonga.com${this.$route.path}`,
        },
      ],
      meta: [
        {
          hid: 'description',
          name: 'description',
          content:
            'In my blog, I share what I have been learning and some projects I have worked on. I mostly write about Data Science, Machine Learning, among other topics I am interested in.',
        },
        // Open Graph
        { hid: 'og:title', property: 'og:title', content: 'Blog' },
        {
          hid: 'og:description',
          property: 'og:description',
          content:
            'In my blog, I share what I have been learning and some projects I have worked on. I mostly write about Data Science, Machine Learning, among other topics I am interested in.',
        },
        {
          hid: 'og:url',
          property: 'og:url',
          content: `https://hmatalonga.com${this.$route.path}`,
        },
        // Twitter Card
        {
          hid: 'twitter:title',
          name: 'twitter:title',
          content: 'Blog',
        },
        {
          hid: 'twitter:description',
          name: 'twitter:description',
          content:
            'In my blog, I share what I have been learning and some projects I have worked on. I mostly write about Data Science, Machine Learning, among other topics I am interested in.',
        },
      ],
    }
  },
}
</script>
