<template>
  <div>
    <h1
      class="text-5xl font-semibold leading-tight text-gray-800 md:text-6xl lg:text-7xl lg:font-medium xl:text-8xl"
    >
      Hi there! 👋
    </h1>
    <p
      class="mt-8 md:mt-10 xl:mt-16 text-lg text-gray-700 md:text-xl lg:text-2xl"
    >
      My name is Hugo Matalonga, and I am a data scientist and a web developer
      based in Portugal.
    </p>
    <div
      class="flex items-center justify-start mt-6 text-xs font-medium text-gray-700 space-x-6 sm:text-sm lg:text-base lg:mt-8"
    >
      <a
        href="mailto:hello@hmatalonga.com"
        class="flex items-center hover:text-gray-800"
        ><svg class="h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
          <title>Email</title>
          <path
            d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z"
          ></path>
          <path
            d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z"
          ></path></svg
        ><span>Email</span></a
      >
      <a
        href="https://twitter.com/hmatalonga"
        class="flex items-center hover:text-gray-800"
        ><svg
          class="fill-current h-4 mr-2"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
        >
          <title>Twitter</title>
          <path
            d="M6.29 18.25c7.55 0 11.67-6.25 11.67-11.67v-.53c.8-.59 1.49-1.3 2.04-2.13-.75.33-1.54.55-2.36.65a4.12 4.12 0 0 0 1.8-2.27c-.8.48-1.68.81-2.6 1a4.1 4.1 0 0 0-7 3.74 11.65 11.65 0 0 1-8.45-4.3 4.1 4.1 0 0 0 1.27 5.49C2.01 8.2 1.37 8.03.8 7.7v.05a4.1 4.1 0 0 0 3.3 4.03 4.1 4.1 0 0 1-1.86.07 4.1 4.1 0 0 0 3.83 2.85A8.23 8.23 0 0 1 0 16.4a11.62 11.62 0 0 0 6.29 1.84"
          ></path></svg
        ><span>Twitter</span></a
      ><a
        href="https://github.com/hmatalonga"
        class="flex items-center hover:text-gray-800"
        ><svg
          class="fill-current h-4 mr-2"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 32.58 31.77"
        >
          <path
            d="M16.29,0C7.29,0,0,7.29,0,16.29c0,7.2,4.67,13.3,11.14,15.46c0.81,0.15,1.11-0.35,1.11-0.79 c0-0.39-0.01-1.41-0.02-2.77c-4.53,0.98-5.49-2.18-5.49-2.18C6,24.13,4.93,23.62,4.93,23.62c-1.48-1.01,0.11-0.99,0.11-0.99 c1.63,0.12,2.5,1.68,2.5,1.68c1.45,2.49,3.81,1.77,4.74,1.35c0.15-1.05,0.57-1.77,1.03-2.18C9.7,23.08,5.9,21.68,5.9,15.44 c0-1.78,0.63-3.23,1.68-4.37C7.41,10.65,6.85,9,7.73,6.76c0,0,1.37-0.44,4.48,1.67c1.3-0.36,2.69-0.54,4.08-0.55 c1.38,0.01,2.78,0.19,4.08,0.55c3.11-2.11,4.48-1.67,4.48-1.67c0.89,2.24,0.33,3.9,0.16,4.31c1.04,1.14,1.67,2.59,1.67,4.37 c0,6.26-3.81,7.63-7.44,8.04c0.58,0.5,1.11,1.5,1.11,3.02c0,2.18-0.02,3.93-0.02,4.47c0,0.44,0.29,0.94,1.12,0.78 c6.47-2.16,11.13-8.26,11.13-15.45C32.58,7.29,25.29,0,16.29,0z"
          ></path></svg
        ><span>GitHub</span></a
      >
    </div>
    <div
      class="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8 mt-10 md:mt-16 lg:mt-20"
    >
      <div>
        <h2 class="text-gray-800 font-semibold text-lg lg:text-xl xl:text-2xl">
          Projects
        </h2>
        <div class="mt-2 md:mt-4 lg:mt-6">
          <p class="text-gray-700 text-base lg:text-lg">
            Currently, I am working in Academia as a Researcher. My main focus
            is on Privacy-Preserving Machine Learning and Transfer Learning with
            tree-based models.
          </p>
        </div>
      </div>
      <div>
        <h2 class="text-gray-800 font-semibold text-lg lg:text-xl xl:text-2xl">
          Blog
        </h2>
        <div class="mt-2 md:mt-4 lg:mt-6">
          <p class="text-gray-700 text-base lg:text-lg">
            In
            <nuxt-link class="text-indigo-600 underline" to="/blog"
              >my blog</nuxt-link
            >, I share what I have been learning and some projects I have worked
            on. I mostly write about Data Science, Machine Learning, among other
            topics I am interested in.
          </p>
        </div>
      </div>
    </div>
  </div>
</template>
